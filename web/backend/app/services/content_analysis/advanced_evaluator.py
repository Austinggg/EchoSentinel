import logging
import requests
import json
import time
import re
import openai
from pathlib import Path
from assessment_evaluator import AssessmentEvaluator

class AdvancedEvaluator:
    """高级判断器，支持知识图谱查询和联网搜索，使用ReAct代理架构"""
    
    def __init__(self, kg_base_url="http://localhost:8887", config=None):
        """初始化高级评估器
          
        Args:
            kg_base_url: 知识图谱服务的基础URL
            config: 可选的配置参数
        """
        self.kg_base_url = kg_base_url
        self.kg_session = requests.Session()
        self.default_project_id = 1
        self.default_user_id = 1
        
        # 加载配置
        if config is None:
            self.config = self._load_config()
        else:
            self.config = config
            
        # 从配置中获取Cookie
        self.kg_cookie = self.config.get('KAG_API', {}).get('cookie', '')
        if not self.kg_cookie:
            logging.warning("未找到知识图谱Cookie，查询可能会失败")
            
        # 从配置中获取重试配置
        retry_config = self.config.get('retry', {})
        self.max_retries = retry_config.get('max_retries', 3)
        self.retry_delay = retry_config.get('retry_delay', 1)
        
        # 初始化基础评估器
        self.base_evaluator = AssessmentEvaluator(self.config)
        
        # 初始化ReAct代理
        self._initialize_react_agent()
        
    def _initialize_react_agent(self):
        """初始化ReAct代理"""
        # 获取OpenAI模型配置
        openai_config = self.config.get('assessment_model', {}).get('remote_openai', {})
        
        if not openai_config:
            logging.warning("未找到OpenAI配置，ReAct代理可能无法正常工作")
            return
        
        # 配置OpenAI客户端
        self.openai_client = openai.OpenAI(
            api_key=openai_config.get('api_key'),
            base_url=openai_config.get('base_url')
        )
        
        self.model = openai_config.get('model')
        logging.info(f"已初始化ReAct代理，使用模型: {self.model}")
    
    def _load_config(self):
        """从默认位置加载配置文件"""
        config_path = Path(__file__).parent / 'config' / 'config.json'
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在于：{config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                # 处理BOM头
                if raw_content.startswith('\ufeff'):
                    raw_content = raw_content[1:]
                return json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误：{e.doc}")
        except Exception as e:
            raise RuntimeError(f"配置加载失败：{str(e)}")
    
    def create_kg_session(self, question, project_id=None, user_id=None):
        """创建一个知识图谱对话会话
        
        Args:
            question: 要评估的问题
            project_id: 项目ID，默认为1
            user_id: 用户ID，默认为1
            
        Returns:
            session_id: 创建的会话ID
        """
        if project_id is None:
            project_id = self.default_project_id
        
        if user_id is None:
            user_id = self.default_user_id
            
        payload = {
            "projectId": project_id,
            "name": question,
            "userId": user_id
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.kg_session.post(
                    f"{self.kg_base_url}/public/v1/reasoner/session/create",
                    json=payload
                )
                
                if response.status_code != 200:
                    logging.error(f"创建会话失败: {response.text}")
                    time.sleep(self.retry_delay)
                    continue
                    
                data = response.json()
                if not data.get("success"):
                    logging.error(f"创建会话失败: {data}")
                    time.sleep(self.retry_delay)
                    continue
                    
                return data["result"]["id"]
            except Exception as e:
                logging.error(f"创建会话异常: {str(e)}")
                time.sleep(self.retry_delay)
        
        raise Exception("创建会话重试次数超限")
    
    def query_knowledge_graph(self, question, session_id, project_id=None, thinking_enabled=True):
        """查询知识图谱工具
        
        Args:
            question: 要查询的问题
            session_id: 会话ID
            project_id: 项目ID，默认为1
            thinking_enabled: 是否启用思考模式
            
        Returns:
            answer: 查询结果
        """
        if project_id is None:
            project_id = self.default_project_id
            
        payload = {
            "prompt": [
                {
                    "type": "text",
                    "content": question
                }
            ],
            "session_id": session_id,
            "project_id": project_id,
            "thinking_enabled": thinking_enabled,
            "search_enabled": False
        }
        
        # 如果有Cookie，添加到请求头中
        headers = {}
        if self.kg_cookie:
            headers['Cookie'] = self.kg_cookie
        
        for attempt in range(self.max_retries):
            try:
                response = self.kg_session.post(
                    f"{self.kg_base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    logging.error(f"查询知识图谱失败: {response.text}")
                    time.sleep(self.retry_delay)
                    continue
                    
                return response.json()
            except Exception as e:
                logging.error(f"查询知识图谱异常: {str(e)}")
                time.sleep(self.retry_delay)
        
        raise Exception("查询知识图谱重试次数超限")
    
    def web_search(self, query):
        """网络搜索工具
        
        Args:
            query: 搜索查询
            
        Returns:
            search_results: 搜索结果
        """
        logging.info(f"执行网络搜索: {query}")
        # TODO: 实现实际的网络搜索功能
        return f"网络搜索结果: 关于'{query}'的信息..."
    
    def react_evaluate(self, question, tools=None):
        """使用ReAct代理进行评估
        
        Args:
            question: 评估问题
            tools: 可用工具列表
            
        Returns:
            评估结果
        """
        if not hasattr(self, 'openai_client') or not self.model:
            logging.error("ReAct代理未正确初始化")
            raise RuntimeError("ReAct代理未正确初始化")
        
        # 默认提供知识图谱和网络搜索工具
        if tools is None:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "query_knowledge_graph",
                        "description": "查询知识图谱获取事实信息",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "要查询的问题"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "通过网络搜索查找信息",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "搜索查询内容"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        
        messages = [
            {
                "role": "system",
                "content": """你是一个评估信息内容的ReAct代理。请遵循以下步骤进行分析：
    1. 思考(Thought)：分析问题，思考需要什么信息来回答
    2. 行动(Action)：决定是否需要使用工具，如知识图谱查询或网络搜索
    3. 观察(Observation)：分析工具返回的结果
    4. 继续思考和行动，直到你有足够信息
    5. 最后给出评估结果和分数(0-1之间，1表示最佳)
    
    请确保你的分析逻辑清晰，最终给出明确的评分和理由。"""
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        # 创建知识图谱会话
        session_id = self.create_kg_session(question)
        
        # ReAct代理交互循环
        max_turns = 5
        for _ in range(max_turns):
            try:
                # 调用模型
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                messages.append(message)
                
                # 检查是否需要调用工具
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        
                        # 修复：安全处理 arguments 参数
                        arguments = {}
                        if hasattr(tool_call.function, 'arguments'):
                            args = tool_call.function.arguments
                            if isinstance(args, dict):
                                arguments = args
                            elif isinstance(args, str):
                                try:
                                    arguments = json.loads(args)
                                except json.JSONDecodeError:
                                    logging.error(f"无法解析函数参数: {args}")
                                    arguments = {"query": question}  # 使用问题作为默认查询
                            else:
                                logging.error(f"未预期的参数类型: {type(args)}")
                                arguments = {"query": question}
                        
                        # 执行工具调用
                        if function_name == "query_knowledge_graph":
                            query = arguments.get("query", question)
                            # 实际调用知识图谱
                            kg_result = self.query_knowledge_graph(
                                query, 
                                session_id=session_id,
                                thinking_enabled=True
                            )
                            # 提取内容
                            kg_content = kg_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                            tool_result = kg_content
                        elif function_name == "web_search":
                            query = arguments.get("query", question)
                            tool_result = self.web_search(query)
                        else:
                            tool_result = f"未知工具: {function_name}"
                        
                        # 添加工具响应到消息列表
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_result
                        })
                else:
                    # 没有工具调用，检查是否有最终答案
                    final_answer = message.content
                    if self._contains_score(final_answer):
                        return self._extract_result(final_answer)
            
            except Exception as e:
                logging.error(f"ReAct评估过程出错: {str(e)}")
                # 添加详细调试信息
                if hasattr(message, 'tool_calls') and message.tool_calls and hasattr(message.tool_calls[0], 'function'):
                    logging.debug(f"函数参数类型: {type(message.tool_calls[0].function.arguments)}")
                    logging.debug(f"函数参数内容: {message.tool_calls[0].function.arguments}")
                break
        
        # 如果达到最大轮次或发生错误，尝试从最后一条消息提取结果
        if messages and len(messages) > 1:
            return self._extract_result(messages[-1].content)
        else:
            return {"score": 0.5, "reasoning": "评估过程未能得出明确结论"}
    
    def _contains_score(self, text):
        """检查文本是否包含评分结果"""
        return bool(re.search(r'(评分|得分|分数|评估结果|最终得分)[:\s]*(\d+(\.\d+)?)', text))
    
    def _extract_result(self, text):
        """从文本中提取评估结果"""
        # 尝试提取分数
        score_match = re.search(r'(评分|得分|分数|评估结果|最终得分)[:\s]*(\d+(\.\d+)?)', text)
        if score_match:
            try:
                score = float(score_match.group(2))
                # 确保分数在0-1范围内
                if score > 1:
                    score = score / 10 if score <= 10 else 0.5
            except:
                score = 0.5
        else:
            # 基于关键词评估
            if any(phrase in text.lower() for phrase in ["高", "好", "充分", "准确", "完整", "没有不当"]):
                score = 0.8
            elif any(phrase in text.lower() for phrase in ["低", "差", "不充分", "不准确", "不完整", "有不当"]):
                score = 0.2
            else:
                score = 0.5
        
        return {
            "score": score,
            "reasoning": text
        }

    def advanced_assessment(self, message, assessment_type, include_reasoning=False):
        """高级评估流程
        
        Args:
            message: 要评估的信息
            assessment_type: 评估类型 (p1-p8)
            include_reasoning: 是否包含推理内容
            
        Returns:
            评估结果
        """
        # 构建评估问题
        question = self._build_question(message, assessment_type)
        
        try:
            # 使用ReAct代理进行评估
            result = self.react_evaluate(question)
            
            if include_reasoning:
                return {
                    "score": result["score"],
                    "reasoning_content": result["reasoning"]
                }
            else:
                return result["score"]
                
        except Exception as e:
            logging.exception(f"高级评估失败: {str(e)}")
            # 回退到基本评估方法
            logging.info("回退到基础评估方法")
            return self._fallback_to_basic_assessment(message, assessment_type, include_reasoning)
    
    def _build_question(self, message, assessment_type):
        """根据评估类型构建查询问题"""
        # 构建包含ReAct框架的评估问题
        assessment_questions = {
            "p1": f"""请使用ReAct框架评估以下消息的背景信息充分性:
"{message}"

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：识别消息中提供的关键背景信息以及可能缺失的信息
2. 行动：查询或搜索相关信息以确定是否存在必要的背景信息
3. 观察：分析查询结果
4. 继续思考和行动直到能给出评估
5. 最终评分：0-1之间的分数(1表示背景信息完全充分)""",

            "p2": f"""请使用ReAct框架评估以下消息的背景信息准确性:
"{message}"

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：识别消息中的关键事实性主张
2. 行动：查询知识图谱或搜索验证这些主张的准确性
3. 观察：分析查询结果
4. 继续思考和行动直到能评估所有主张
5. 最终评分：0-1之间的分数(1表示完全准确)""",

            "p3": f"""请使用ReAct框架评估以下消息是否有内容被故意删除导致意思被歪曲:
"{message}"

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：检查消息的连贯性和完整性
2. 行动：查询或搜索相关完整信息
3. 观察：分析是否存在逻辑跳跃或不连贯的部分
4. 继续思考和行动直到能给出评估
5. 最终评分：0-1之间的分数(1表示内容完全完整)""",

            "p4": f"""请使用ReAct框架评估以下消息是否存在不当意图:
"{message}"

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：分析消息可能的意图类型
2. 行动：查询相关规范和标准
3. 观察：将消息与标准对比
4. 继续思考和行动直到能给出评估
5. 最终评分：0-1之间的分数(1表示没有不当意图)""",

            "p5": f"""请使用ReAct框架评估发布者历史记录:
发布者信誉度: {message}

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：分析信誉度数据的含义
2. 行动：查询类似信誉度下的发布者行为模式
3. 观察：分析这些模式对内容可信度的影响
4. 继续思考和行动直到能给出评估
5. 最终评分：0-1之间的分数(1表示信誉良好)""",

            "p6": f"""请使用ReAct框架评估以下消息的情感煽动性:
"{message}"

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：识别消息中的情感性语言和修辞手法
2. 行动：查询此类语言的煽动性标准
3. 观察：评估消息的整体情感基调
4. 继续思考和行动直到能给出评估
5. 最终评分：0-1之间的分数(1表示完全情感中立)""",

            "p7": f"""请使用ReAct框架评估以下消息是否包含诱导行为:
"{message}"

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：识别消息中的号召性用语或行动建议
2. 行动：查询这类用语的诱导性标准
3. 观察：评估这些建议的紧迫性和强制性
4. 继续思考和行动直到能给出评估
5. 最终评分：0-1之间的分数(1表示完全没有诱导行为)""",

            "p8": f"""请使用ReAct框架评估以下陈述列表的信息一致性:
{message}

请按ReAct框架的思考-行动-观察模式分析:
1. 思考：提取每个陈述中的关键事实和主张
2. 行动：查询相关信息验证事实一致性
3. 观察：比较不同陈述之间的事实是否一致
4. 继续思考和行动直到能给出评估
5. 最终评分：0-1之间的分数(1表示完全一致)"""
        }
        
        return assessment_questions.get(assessment_type, f"请使用ReAct框架评估这条消息：{message}")
    
    def _fallback_to_basic_assessment(self, message, assessment_type, include_reasoning=False):
        """当高级评估失败时回退到基本评估方法"""
        logging.info(f"回退到基本评估，类型: {assessment_type}")
        
        try:
            # 根据评估类型调用不同的基本评估方法
            if assessment_type == "p1":
                return self.base_evaluator.p1_assessment(message, include_reasoning=include_reasoning)
            elif assessment_type == "p2":
                return self.base_evaluator.p2_assessment(message, include_reasoning=include_reasoning)
            elif assessment_type == "p3":
                return self.base_evaluator.p3_assessment(message, include_reasoning=include_reasoning)
            elif assessment_type == "p4":
                # 这里需要意图参数，但高级评估中没有传入，使用空列表代替
                return self.base_evaluator.p4_assessment(message, [], include_reasoning=include_reasoning)
            elif assessment_type == "p5":
                # 使用默认的声誉值
                return self.base_evaluator.p5_assessment(0.5, include_reasoning=include_reasoning)
            elif assessment_type == "p6":
                return self.base_evaluator.p6_assessment(message, include_reasoning=include_reasoning)
            elif assessment_type == "p7":
                return self.base_evaluator.p7_assessment(message, include_reasoning=include_reasoning)
            elif assessment_type == "p8":
                # p8可能需要陈述列表，这里假设message是列表形式
                statements = message if isinstance(message, list) else [message]
                return self.base_evaluator.p8_assessment(statements, include_reasoning=include_reasoning)
            else:
                # 未知的评估类型，使用p1作为默认
                logging.warning(f"未知的评估类型: {assessment_type}，使用p1评估")
                return self.base_evaluator.p1_assessment(message, include_reasoning=include_reasoning)
        except Exception as e:
            logging.exception(f"基本评估回退也失败: {str(e)}")
            # 最终回退方案：返回中间值
            if include_reasoning:
                return {
                    "score": 0.5,
                    "reasoning_content": f"评估过程出错: {str(e)}"
                }
            else:
                return 0.5
    
    # 为8个评估项添加专门方法，方便直接调用
    def p1_assessment(self, message, include_reasoning=False):
        """背景信息充分性评估"""
        return self.advanced_assessment(message, "p1", include_reasoning)
    
    def p2_assessment(self, message, include_reasoning=False):
        """背景信息准确性评估"""
        return self.advanced_assessment(message, "p2", include_reasoning)
    
    def p3_assessment(self, message, include_reasoning=False):
        """内容完整性评估"""
        return self.advanced_assessment(message, "p3", include_reasoning)
    
    def p4_assessment(self, message, intent=None, include_reasoning=False):
        """不当意图评估"""
        # 如果提供了意图，将其添加到消息中
        if intent and isinstance(intent, list) and len(intent) > 0:
            message = f"{message}\n意图: {', '.join(intent)}"
        return self.advanced_assessment(message, "p4", include_reasoning)
    
    def p5_assessment(self, reputation=0.5, include_reasoning=False):
        """发布者历史评估"""
        message = f"发布者信誉度: {reputation}"
        return self.advanced_assessment(message, "p5", include_reasoning)
    
    def p6_assessment(self, message, include_reasoning=False):
        """情感煽动性评估"""
        return self.advanced_assessment(message, "p6", include_reasoning)
    
    def p7_assessment(self, message, include_reasoning=False):
        """诱导行为评估"""
        return self.advanced_assessment(message, "p7", include_reasoning)
    
    def p8_assessment(self, statements, include_reasoning=False):
        """信息一致性评估"""
        # 将陈述列表转换为文本格式
        if isinstance(statements, list):
            message = "\n".join([f"{i+1}. {stmt}" if isinstance(stmt, str) else f"{i+1}. {stmt.get('content', '')}" 
                                for i, stmt in enumerate(statements)])
        else:
            message = str(statements)
        
        return self.advanced_assessment(message, "p8", include_reasoning)

# 示例用法
def example_usage():
    # 初始化评估器
    evaluator = AdvancedEvaluator(kg_base_url="http://localhost:8887")
    
    # 示例消息
    message = "某品牌牛奶被检测出致癌物！转发超过10次可领取现金红包。"
    
    try:
        # 带推理的评估
        result = evaluator.p2_assessment(message, include_reasoning=True)
        print(f"信息准确性评分: {result['score']}")
        print(f"推理过程:\n{result['reasoning_content']}")
        
        # 不带推理的评估
        score = evaluator.p6_assessment(message)
        print(f"情感煽动性评分: {score}")
    except Exception as e:
        print(f"评估失败: {str(e)}")

# 如果直接运行此文件，执行示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()