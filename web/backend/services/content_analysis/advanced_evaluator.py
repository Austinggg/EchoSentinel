import logging
import requests
import json
import time
from pathlib import Path
from assessment_evaluator import AssessmentEvaluator

class AdvancedEvaluator:
    """高级判断器，支持知识图谱查询和联网搜索"""
    
    def __init__(self, base_url="http://localhost:8887", config=None):
        """初始化高级评估器
        
        Args:
            base_url: 知识图谱服务的基础URL
            config: 可选的配置参数
        """
        self.base_url = base_url
        self.session = requests.Session()
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
    
    def create_session(self, question, project_id=None, user_id=None):
        """创建一个对话会话
        
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
                response = self.session.post(
                    f"{self.base_url}/public/v1/reasoner/session/create",
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
        """查询知识图谱
        
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
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers  # 添加Cookie
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
            # 步骤1: 创建会话
            session_id = self.create_session(question)
            logging.info(f"已创建会话 ID: {session_id}")
            
            # 步骤2: 简单问答模式
            logging.info("开始简单问答模式")
            result = self.query_knowledge_graph(
                question, 
                session_id, 
                thinking_enabled=False
            )
            
            # 检查是否需要进一步查询
            if self._needs_further_query(result):
                logging.info("简单模式无法回答，切换到思考模式")
                # 步骤3: 开启thinking模式查询知识图谱
                result = self.query_knowledge_graph(
                    question,
                    session_id,
                    thinking_enabled=True
                )
                
                # 未来支持的功能: 联网搜索
                if self._needs_further_query(result):
                    logging.info("思考模式无法回答，将来会支持联网搜索")
                    # TODO: 联网搜索功能
            
            # 解析结果
            score, reasoning = self._parse_result(result)
            
            if include_reasoning:
                return {
                    "score": score,
                    "reasoning_content": reasoning
                }
            else:
                return score
                
        except Exception as e:
            logging.exception(f"高级评估失败: {str(e)}")
            # 回退到基本评估方法
            logging.info("回退到基础评估方法")
            return self._fallback_to_basic_assessment(message, assessment_type, include_reasoning)
    
    def _build_question(self, message, assessment_type):
        """根据评估类型构建查询问题"""
        # 基础评估 (不需要知识图谱的项目)
        basic_assessment = {
            "p1": f"""请评估以下消息的背景信息充分性:
    "{message}"

    请按以下步骤进行分析:
    1. 识别消息中提供的关键背景信息
    2. 分析这些信息是否足够理解消息的上下文
    3. 指出消息中缺失的重要背景信息
    4. 总结评估并给出0-1之间的分数(1表示背景信息完全充分)""",

            "p3": f"""请评估以下消息是否有内容被故意删除导致意思被歪曲:
    "{message}"

    请按以下步骤进行分析:
    1. 检查消息的连贯性和完整性
    2. 寻找可能表明信息被删减的线索
    3. 分析是否存在逻辑跳跃或不连贯的部分
    4. 总结评估并给出0-1之间的分数(1表示内容完全完整)""",

            "p6": f"""请评估以下消息的情感煽动性:
    "{message}"

    请按以下步骤进行分析:
    1. 识别消息中的情感性语言和修辞手法
    2. 分析这些表达是否旨在激发读者的强烈情感反应
    3. 评估消息的整体情感基调
    4. 总结评估并给出0-1之间的分数(1表示完全情感中立)""",

            "p7": f"""请评估以下消息是否包含诱导行为:
    "{message}"

    请按以下步骤进行分析:
    1. 识别消息中的号召性用语或行动建议
    2. 分析这些建议是如何呈现的(命令式、暗示式等)
    3. 评估这些建议的紧迫性和强制性
    4. 总结评估并给出0-1之间的分数(1表示完全没有诱导行为)""",

            "p8": f"""请评估以下陈述列表的信息一致性:
    {message}

    请按以下步骤进行分析:
    1. 提取每个陈述中的关键事实和主张
    2. 比较不同陈述之间的事实是否一致
    3. 指出任何矛盾或不一致的信息
    4. 总结评估并给出0-1之间的分数(1表示完全一致)"""
        }
        # 需要知识图谱的高级评估
        advanced_assessment = {
            "p2": f"""请使用ReAct方法评估以下消息的背景信息准确性:
    "{message}"

    步骤:
    1. 思考: 识别消息中的关键事实性主张
    2. 行动: 查询这些主张的准确性
    3. 观察: 分析查询结果
    4. 思考: 评估主张与事实的符合程度
    5. 结论: 给出最终评分(0-1,1表示完全准确)""",

            "p4": f"""请使用ReAct方法评估以下消息是否存在不当意图:
    "{message}"

    步骤:
    1. 思考: 分析消息可能的意图类型
    2. 行动: 查询相关规范和标准
    3. 观察: 将消息与标准对比
    4. 思考: 评估意图的适当性
    5. 结论: 给出最终评分(0-1,1表示没有不当意图)""",

            "p5": f"""请使用ReAct方法评估发布者历史记录:
    发布者信誉度: {message}

    步骤:
    1. 思考: 分析信誉度数据的含义
    2. 行动: 查询类似信誉度下的发布者行为模式
    3. 观察: 分析这些模式对内容可信度的影响
    4. 思考: 评估整体信誉状况
    5. 结论: 给出最终评分(0-1,1表示信誉良好)"""
        }
            # 根据评估类型选择适当的提示
        if assessment_type in advanced_assessment and self.config.get('use_knowledge_graph', True):
            return advanced_assessment.get(assessment_type)
        else:
            return basic_assessment.get(assessment_type, f"请评估这条消息：{message}")
    def advanced_assessment(self, message, assessment_type, include_reasoning=False):
        """采用ReAct框架的高级评估流程"""
        
        # 判断是否需要知识图谱查询
        needs_kg = assessment_type in ["p2", "p4", "p5"] and self.config.get('use_knowledge_graph', True)
        
        # 构建评估问题
        question = self._build_question(message, assessment_type)
        
        try:
            # 如果不需要知识图谱，直接使用基础评估器
            if not needs_kg:
                return self._fallback_to_basic_assessment(message, assessment_type, include_reasoning)
            
            # 创建会话
            session_id = self.create_session(question)
            logging.info(f"已创建会话 ID: {session_id}")
            
            # 执行ReAct风格的查询
            logging.info("开始ReAct评估流程")
            result = self.query_knowledge_graph(
                question, 
                session_id, 
                thinking_enabled=True  # ReAct需要启用思考模式
            )
            
            # 如果结果不满意，尝试联网搜索(未来功能)
            if self._needs_further_query(result):
                logging.info("知识图谱无法回答，将来会支持联网搜索")
                # TODO: 联网搜索功能
            
            # 解析结果
            score, reasoning = self._parse_result(result)
            
            if include_reasoning:
                return {
                    "score": score,
                    "reasoning_content": reasoning
                }
            else:
                return score
                
        except Exception as e:
            logging.exception(f"高级评估失败: {str(e)}")
            # 回退到基本评估方法
            logging.info("回退到基础评估方法")
            return self._fallback_to_basic_assessment(message, assessment_type, include_reasoning)
    def _needs_further_query(self, result):
        """判断是否需要进一步查询"""
        # 获取返回内容
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 检查是否包含不确定性表达
        uncertainty_phrases = [
            "我不知道", "无法确定", "不确定", "无法评估",
            "没有足够信息", "信息不足", "无法判断", "无法给出"
        ]
        
        return any(phrase in content for phrase in uncertainty_phrases)
    
    def _parse_result(self, result):
        """解析ReAct格式的查询结果，提取分数和推理过程"""
        # 提取内容
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        thinking = result.get("choices", [{}])[0].get("thinking", "")
        
        # 尝试从内容最后部分提取分数
        score = None
        try:
            # 寻找结论部分
            conclusion_sections = ["结论:", "总结评估", "最终评分", "最终得分", "评分结果"]
            conclusion_text = ""
            
            for section in conclusion_sections:
                if section in content:
                    parts = content.split(section)
                    if len(parts) > 1:
                        conclusion_text = parts[1].strip()
                        break
            
            # 如果找到结论部分，从中提取分数
            if conclusion_text:
                score_match = re.search(r'(\d+(\.\d+)?)', conclusion_text)
                if score_match:
                    score_str = score_match.group(1)
                    score = float(score_str)
                    # 确保分数在0-1范围内
                    if score > 1:
                        score = score / 10 if score <= 10 else 0.5
            
            # 如果仍然无法提取，检查整个内容
            if score is None:
                score_match = re.search(r'(\d+(\.\d+)?)', content)
                if score_match:
                    score_str = score_match.group(1)
                    score = float(score_str)
                    if score > 1:
                        score = score / 10 if score <= 10 else 0.5
            
            # 如果还是无法提取，基于关键词评估
            if score is None:
                if any(phrase in content.lower() for phrase in ["高", "好", "充分", "准确", "完整", "没有不当"]):
                    score = 0.8
                elif any(phrase in content.lower() for phrase in ["低", "差", "不充分", "不准确", "不完整", "有不当"]):
                    score = 0.2
                else:
                    score = 0.5
        except Exception as e:
            logging.error(f"解析分数失败: {str(e)}")
            score = 0.5  # 默认中间值
        
        # 组合思考过程和内容作为推理过程，突出ReAct步骤
        reasoning = ""
        if thinking:
            reasoning += f"思考过程:\n{thinking}\n\n"
        
        # 美化内容展示
        formatted_content = content
        react_steps = ["思考:", "行动:", "观察:", "结论:"]
        for step in react_steps:
            if step in formatted_content:
                formatted_content = formatted_content.replace(step, f"\n## {step}")
        
        reasoning += f"分析过程:\n{formatted_content}"
        
        return score, reasoning
    
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
    
    def web_search(self, question, session_id, project_id=None):
        """进行网络搜索（预留接口，未实现）
        
        Args:
            question: 要搜索的问题
            session_id: 会话ID
            project_id: 项目ID，默认为1
            
        Returns:
            answer: 搜索结果
        """
        logging.warning("网络搜索功能尚未实现，将在未来版本添加")
        # TODO: 实现网络搜索功能
        return {
            "choices": [
                {
                    "message": {
                        "content": "网络搜索功能尚未实现，无法提供结果。"
                    }
                }
            ]
        }
    
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
    evaluator = AdvancedEvaluator(base_url="http://localhost:8887")
    
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