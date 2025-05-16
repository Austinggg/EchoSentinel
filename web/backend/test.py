import os
import sys

# sys.path.append("src")
from opendeepsearch import OpenDeepSearchTool
from smolagents import LiteLLMModel, ToolCallingAgent

# 设置API密钥
os.environ["SERPER_API_KEY"] = "d2406b655457ab10b1bd94a1469ea1ce42a78984"  # Serper
os.environ["JINA_API_KEY"] = "jina_a5fe318b0e0148a7b1b84f9d11bb2911ngUuEpvXV6rUeFVOfTgX8m6B6Qv2"
os.environ['DEEPSEEK_API_KEY'] = "sk-0f87344ec4d74b9ebf4e8dad6b7dbb47"
# 用 LiteLLMModel 代替 openai.OpenAI client
model = LiteLLMModel(
    "deepseek/deepseek-chat",  # deepseek 官方模型名
    temperature=0.2,
    provider="deepseek"
)

# 初始化搜索工具
search_agent = OpenDeepSearchTool(
    model_name="deepseek/deepseek-chat",
    reranker="jina",
    search_provider="serper",
    serper_api_key="d2406b655457ab10b1bd94a1469ea1ce42a78984"
)

# 创建ToolCallingAgent
tool_agent = ToolCallingAgent(
    tools=[search_agent],
    model=model,
)

# 运行查询
query = """
这条消息是真的吗？
时间：2014-03-16
消息内容：
摸奶节是中国云南双柏县鄂家镇彝族传统文化的庆典就是农历的7月14日、15日与16日这三天，包括外来的游人，如果在街上遇见喜欢的女子，都可以摸一摸女子的胸部。姑娘们表面躲躲闪闪，但决无责怪之意因为这是他们这个地区的百姓延续了1000多年的风俗。小伙子以摸到奶为吉祥，姑娘们以被摸而高兴。
"""
result = tool_agent.run(query)
search_result = search_agent.forward(query)
print("搜索结果:")
print(search_result)
# print("\n最终回答:")
# print(result)