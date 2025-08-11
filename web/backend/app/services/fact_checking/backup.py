import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from smolagents import LiteLLMModel, ToolCallingAgent
from app.services.fact_checking.AISearchTool import AISearchTool

# 配置日志
logger = logging.getLogger(__name__)

class FactCheckPipeline:
    """
    事实核查处理流水线，包含三个主要步骤：
    1. 判断文本是否值得核查
    2. 提取高价值断言
    3. 对断言进行事实核查
    """
    
    def __init__(self, model_name="deepseek/deepseek-chat", 
                 api_url="http://127.0.0.1:8000",
                 api_key=None):
        """初始化事实核查流水线"""
        # 如果提供了API密钥，则使用它
        if api_key:
            os.environ['DEEPSEEK_API_KEY'] = api_key
        elif 'DEEPSEEK_API_KEY' not in os.environ:
            os.environ['DEEPSEEK_API_KEY'] = "sk-0f87344ec4d74b9ebf4e8dad6b7dbb47"  # 默认密钥
            
        try:
            # 初始化基础模型
            self.model = LiteLLMModel(
                model_name,  
                temperature=0.2,
                provider="deepseek"
            )
            
            # 初始化搜索工具
            self.search_agent = AISearchTool(api_url=api_url)
            
            # 创建工具调用代理
            self.tool_agent = ToolCallingAgent(
                tools=[self.search_agent],
                model=self.model,
            )
        except Exception as e:
            logger.error(f"初始化事实核查流水线失败: {str(e)}")
            raise
    
    def is_worth_checking(self, text: str) -> Tuple[bool, str]:
        """判断文本是否值得事实核查"""
        try:
            prompt = f"""
            你是一个专业的内容审核与事实核查助手。

            请判断：这段文本是否"值得进行事实核查"。

            判断标准：
            - 如果文本中存在以下任意情况，请返回"是"，否则返回"否"：
              1. 提及了具体商品、成分、国家、平台等，并声称其具有特殊功效、优势或真实性保障。
              2. 涉及医学、健康、美容、减肥等敏感领域的效果承诺。
              3. 涉及公众信任议题，如国家监管、平台真实性、社会秩序。
              4. 存在虚假宣传嫌疑，如夸张描述、片面承诺、误导性引导。

            请根据以上标准，判断下面这段文本是否值得进行事实核查，并简要说明理由。

            只需返回"是" 或 "否"，然后另起一行给出简要理由。

            文本内容：
            {text}
            """
            
            # 使用run方法获取响应
            response = self.tool_agent.run(prompt)
            result = response.lower()
            
            # 解析结果
            if result.startswith("是"):
                worth_checking = True
                reason = result.split("是", 1)[-1].strip()
            elif result.startswith("否"):
                worth_checking = False
                reason = result.split("否", 1)[-1].strip()
            else:
                # 默认处理
                worth_checking = "是" in result[:10]
                reason = result
            
            # 清理原因文本
            if reason.startswith("：") or reason.startswith(":"):
                reason = reason[1:].strip()
            
            return worth_checking, reason
        except Exception as e:
            logger.error(f"判断文本是否值得核查时出错: {str(e)}")
            return False, f"处理错误: {str(e)}"
    
    def extract_claims(self, text: str) -> List[str]:
        """提取值得核查的断言"""
        try:
            prompt = f"""
            请从以下文本中提取所有"具有事实核查价值"的关键断言（claim）。

            提取标准：
            - 该句子具有明确的事实指向（如某种成分、功效、国家行为、监管政策等）
            - 确保每个断言包含充分的上下文信息（时间、地点、人物、事件）
            - 如果原文中的断言不完整，请适当补充相关上下文使其完整
            - 如果断言涉及的是同一事件的不同方面，尽量将其整合为更完整的断言
            - 每条编号列出，建议不超过4条

            文本内容：
            {text}
            
            请直接列出提取的断言，每条断言前加上序号（如"1."，"2."）。
            确保每个断言都是完整的、可独立核查的，包含足够的上下文信息。
            如果没有可提取的断言，请返回"无"。
            """
            
            # 使用run方法获取响应
            response = self.tool_agent.run(prompt)
            
            if "无" == response.strip():
                return []
                
            # 解析提取的断言
            claims = []
            for line in response.split('\n'):
                line = line.strip()
                # 匹配常见编号格式，如"1."、"1、"、"1："、"（1）"等
                if line and (line[0].isdigit() or 
                             (len(line) > 1 and line[0] == '(' and line[1].isdigit())):
                    # 去除编号部分
                    claim = line.split('.', 1)[-1].split('、', 1)[-1].split('：', 1)[-1].split(':', 1)[-1]
                    claim = claim.strip()
                    if claim:
                        claims.append(claim)
            
            return claims
        except Exception as e:
            logger.error(f"提取断言时出错: {str(e)}")
            return []
    
    def fact_check_claims(self, claims: List[str], context: str = None, original_text: str = None) -> List[Dict]:
        """对断言进行事实核查"""
        results = []
        
        for claim in claims:
            try:
                # 构造查询提示
                query = f"""
                请核实以下断言是否属实：
                
                断言：{claim}
                
                {f'时间背景：{context}' if context else ''}
                
                原始文本上下文：
                {original_text}
                
                请基于可靠的网络搜索结果，客观判断该断言是否属实，并给出支持证据。
                请以"是"或"否"开头回答，表明该断言是否属实，然后另起一行给出详细解释。
                
                在搜索时，请同时搜索断言本身和相关的关键事实，而不仅仅是断言中的个别词语。
                如果搜索结果不足以确定断言的真假，请明确说明证据不足，不要仅基于常识做出判断。
                """
                
                # 使用工具代理进行最终判断
                result = self.tool_agent.run(query)
                
                # 分析结论和处理详细解释
                if result.lower().startswith("是"):
                    is_true = "是"
                    # 移除开头的"是"并分离详细解释
                    explanation = result[1:].strip()
                    if explanation.startswith("：") or explanation.startswith(":"):
                        explanation = explanation[1:].strip()
                elif result.lower().startswith("否"):
                    is_true = "否"
                    # 移除开头的"否"并分离详细解释
                    explanation = result[2:].strip()
                    if explanation.startswith("：") or explanation.startswith(":"):
                        explanation = explanation[1:].strip()
                else:
                    is_true = "未确定"
                    explanation = result
                    
                # 整理结果
                fact_check_result = {
                    "claim": claim,
                    "is_true": is_true,
                    "conclusion": explanation  # 使用处理后的解释文本
                }
                
                results.append(fact_check_result)
            except Exception as e:
                logger.error(f"核查断言 '{claim}' 时出错: {str(e)}")
                results.append({
                    "claim": claim,
                    "is_true": "错误",
                    "conclusion": f"核查过程中出错: {str(e)}"
                })
        
        return results
    
    def run_pipeline(self, text: str, context: str = None) -> Dict[str, Any]:
        """运行完整的事实核查流水线"""
        try:
            # 初始化默认结果变量
            search_results = []
            search_keywords = ""
            search_grade = 0
            search_final_answer = ""
            
            # 第一步：判断是否值得核查
            worth_checking, reason = self.is_worth_checking(text)
            
            # 如果不值得核查，提前返回结果
            if not worth_checking:
                return {
                    "worth_checking": False,
                    "reason": reason,
                    "claims": [],
                    "fact_check_results": [],
                    "search_results": search_results,
                    "search_keywords": search_keywords,
                    "search_grade": search_grade,
                    "search_final_answer": search_final_answer
                }
            
            # 第二步：提取高价值断言
            claims = self.extract_claims(text)
            
            if not claims:
                return {
                    "worth_checking": True,
                    "reason": reason,
                    "claims": [],
                    "fact_check_results": [],
                    "search_results": search_results,
                    "search_keywords": search_keywords,
                    "search_grade": search_grade,
                    "search_final_answer": search_final_answer
                }
            
            # 第三步：对断言进行事实核查
            fact_check_results = []
            # 保存最后一个有效的搜索结果
            last_valid_search_data = {}
            
            # 遍历断言进行核查
            for claim in claims:
                try:
                    # 保存当前断言的搜索结果
                    claim_search_results = {}
                    
                    # 构造查询提示
                    search_prompt = f"""
                    请搜索以下断言的相关信息：
                    {claim}
                    
                    {context if context else ''}
                    """
                    # 调用工具进行搜索
                    tool_result = self.tool_agent.run(search_prompt)
                    
                    # 尝试提取搜索结果
                    try:
                        if hasattr(self.search_agent, 'last_search_data'):
                            last_data = self.search_agent.last_search_data
                            
                            # 检查 last_data 类型并妥善处理
                            if isinstance(last_data, dict):
                                # 保存当前断言的有效搜索结果
                                last_valid_search_data = last_data
                                
                                # 保存到当前断言结果中
                                claim_search_results = last_data
                            elif isinstance(last_data, str):
                                # 如果是字符串，尝试解析为字典
                                logger.warning(f"搜索结果是字符串类型，尝试解析为JSON: {last_data[:100]}...")
                                try:
                                    import json
                                    parsed_data = json.loads(last_data)
                                    if isinstance(parsed_data, dict):
                                        last_valid_search_data = parsed_data
                                        claim_search_results = parsed_data
                                    else:
                                        logger.warning(f"解析后的数据不是字典: {type(parsed_data)}")
                                except json.JSONDecodeError:
                                    logger.warning("无法将字符串解析为JSON")
                                    # 创建包含原始字符串的字典
                                    last_valid_search_data = {"text": last_data}
                                    claim_search_results = last_valid_search_data
                            else:
                                logger.warning(f"断言 '{claim}' 的搜索结果类型意外: {type(last_data)}")
                                # 创建包含类型信息的字典
                                last_valid_search_data = {"type": str(type(last_data)), "data": str(last_data)}
                                claim_search_results = last_valid_search_data
                    except Exception as extract_err:
                        logger.warning(f"提取断言 '{claim}' 的搜索结果失败: {str(extract_err)}")
                        logger.debug("异常详情:", exc_info=True)
                    # 进行断言核查，使用工具结果
                    check_prompt = f"""
                    基于以下搜索结果，请核查这个断言是否属实：
                    
                    断言：{claim}
                    
                    搜索结果：
                    {tool_result}
                    
                    请仅回答"是"或"否"，然后解释您的判断。
                    """
                    
                    # 直接使用模型而不是工具代理进行判断
                    check_result = self.model.generate(check_prompt)
                    
                    # 判断结果
                    if isinstance(check_result, str):
                        if check_result.lower().startswith("是"):
                            is_true = "是"
                        elif check_result.lower().startswith("否"):
                            is_true = "否" 
                        else:
                            is_true = "未确定"
                    else:
                        is_true = "错误"
                        check_result = str(check_result)
                    
                    # 添加到结果中
                    fact_check_results.append({
                        "claim": claim,
                        "is_true": is_true,
                        "conclusion": check_result
                    })
                    
                except Exception as claim_err:
                    logger.error(f"处理断言 '{claim}' 时出错: {str(claim_err)}")
                    logger.debug(f"异常详情:", exc_info=True)
                    fact_check_results.append({
                        "claim": claim,
                        "is_true": "错误",
                        "conclusion": f"处理出错: {str(claim_err)}"
                    })
            
            # 提取最后有效搜索数据中的信息
            if isinstance(last_valid_search_data, dict):
                # 更新搜索结果
                if "rerank_results" in last_valid_search_data and isinstance(last_valid_search_data["rerank_results"], list):
                    search_results = last_valid_search_data["rerank_results"]
                if "keywords" in last_valid_search_data:
                    search_keywords = last_valid_search_data["keywords"]
                if "grade" in last_valid_search_data:
                    search_grade = last_valid_search_data["grade"]
                if "final_answer" in last_valid_search_data:
                    search_final_answer = last_valid_search_data["final_answer"]
            
            # 返回完整结果，确保所有字段安全
            return {
                "worth_checking": True,
                "reason": reason,
                "claims": claims,
                "fact_check_results": fact_check_results,
                "search_results": search_results if isinstance(search_results, list) else [],
                "search_keywords": search_keywords if isinstance(search_keywords, str) else "",
                "search_grade": search_grade if isinstance(search_grade, (int, float)) else 0,
                "search_final_answer": search_final_answer if isinstance(search_final_answer, str) else ""
            }
            
        except Exception as e:
            logger.error(f"运行事实核查流水线时出错: {str(e)}")
            logger.debug(f"异常详情:", exc_info=True)
            # 确保错误情况也返回有效结果
            return {
                "error": str(e),
                "worth_checking": False,
                "reason": f"处理出错: {str(e)}",
                "claims": [],
                "fact_check_results": [],
                "search_results": [],
                "search_keywords": "",
                "search_grade": 0,
                "search_final_answer": ""
            }

# 创建单例实例供API使用
_fact_checker = None

def get_fact_checker(model_name="deepseek/deepseek-chat", api_url="http://127.0.0.1:8000"):
    """获取事实核查器实例（单例模式）"""
    global _fact_checker
    if _fact_checker is None:
        try:
            _fact_checker = FactCheckPipeline(model_name=model_name, api_url=api_url)
        except Exception as e:
            logger.error(f"创建事实核查器实例失败: {str(e)}")
            raise
    return _fact_checker