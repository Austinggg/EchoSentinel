import os
from typing import List, Dict, Any, Tuple
from smolagents import LiteLLMModel, ToolCallingAgent
from web.backend.services.fact_checking.AISearchTool import AISearchTool

# 设置API密钥
os.environ['DEEPSEEK_API_KEY'] = "sk-0f87344ec4d74b9ebf4e8dad6b7dbb47"

class FactCheckPipeline:
    """
    事实核查处理流水线，包含三个主要步骤：
    1. 判断文本是否值得核查
    2. 提取高价值断言
    3. 对断言进行事实核查
    """
    
    def __init__(self, model_name="deepseek/deepseek-chat", 
                 api_url="http://127.0.0.1:8000"):
        """初始化事实核查流水线"""
        # 初始化基础模型
        self.model = LiteLLMModel(
            model_name,  
            temperature=0.2,
            provider="deepseek"
        )
        
        # 初始化搜索工具 - 使用自定义的 AISearchTool
        self.search_agent = AISearchTool(api_url=api_url)
        
        # 创建工具调用代理
        self.tool_agent = ToolCallingAgent(
            tools=[self.search_agent],
            model=self.model,
        )
        
    def is_worth_checking(self, text: str) -> Tuple[bool, str]:
        """
        第一步：判断文本是否值得事实核查
        """
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
        
        # 解析结果 - 基于"是"或"否"进行判断
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
    
    def extract_claims(self, text: str) -> List[str]:
        """
        第二步：从值得核查的文本中提取高价值断言
        """
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
    
    def fact_check_claims(self, claims: List[str], context: str = None, original_text: str = None) -> List[Dict]:
        """
        第三步：使用搜索工具对断言进行事实核查
        """
        results = []
        
        for claim in claims:
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
        
        return results
    
    def run_pipeline(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        运行完整的事实核查流水线
        """
        # 第一步：判断是否值得核查
        worth_checking, reason = self.is_worth_checking(text)
        
        # 如果不值得核查，提前返回结果
        if not worth_checking:
            print(f"判定结果：否")
            print(f"理由：{reason}")
            return {
                "worth_checking": False,
                "reason": reason,
                "claims": [],
                "fact_check_results": []
            }
        
        print(f"判定结果：是")
        print(f"理由：{reason}")
        
        # 第二步：提取高价值断言
        claims = self.extract_claims(text)
        
        if not claims:
            print("未提取到有效断言")
            return {
                "worth_checking": True,
                "reason": reason,
                "claims": [],
                "fact_check_results": []
            }
        
        # 第三步：对断言进行事实核查
        fact_check_results = self.fact_check_claims(claims, context, original_text=text)
        
        # 返回完整结果
        return {
            "worth_checking": True,
            "reason": reason,
            "claims": claims,
            "fact_check_results": fact_check_results
        }

# 示例使用
if __name__ == "__main__":
    # 初始化流水线
    pipeline = FactCheckPipeline()
    
    # 测试文本
    test_text = """
女性吃什么能越吃越白？比如有人可能会推荐牛奶和柠檬，但实际上这些方法都是浪费流量，效果不明显。每晚睡前吃点东西能让皮肤更亮更美。用过之后你会爱上它，尤其是脸不漂亮、皮肤可能变黄的女性。所以每晚睡前吃一粒，坚持一段时间后，皮肤会变得晶莹剔透。虽然我们现在都是年轻人，可以不服用任何营养补充剂，但最好每晚睡前服用两粒这种烟酰胺，因为烟酰胺具有很强的还原性，能缓解表皮中的黑色素，从而防止黑色素生成。因为皮肤白净会让人感觉干净漂亮，所以有条件的话姐妹们不妨备一份。但要记住白天不要吃，因为皮肤恢复的最佳时间是在晚上，坚持一个月后你会发现会有显著的变化，这也能为你节省很多保养费用。记住这句话：白能遮丑，黑可真不好看。我在视频左下角放了一个链接。    
""" 
    # 运行流水线
    result = pipeline.run_pipeline(test_text, context="时间：2025-05-16")
    
    # 打印结果
    if result["worth_checking"]:
        print("\n提取的断言:")
        for i, claim in enumerate(result['claims'], 1):
            print(f"{i}. {claim}")
        
        print("\n核查结果:")
        for i, check in enumerate(result['fact_check_results'], 1):
            print(f"\n断言 {i}: {check['claim']}")
            print(f"是否属实: {check['is_true']}")
            print(f"详细解释: {check['conclusion']}")
    else:
        print("\n该内容无需事实核查")