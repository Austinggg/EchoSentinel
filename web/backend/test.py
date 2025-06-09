import os
import sys
from opendeepsearch import OpenDeepSearchAgent, OpenDeepSearchTool
from smolagents import LiteLLMModel, ToolCallingAgent

# 设置API密钥
os.environ["SERPER_API_KEY"] = "d2406b655457ab10b1bd94a1469ea1ce42a78984"  # Serper
os.environ["JINA_API_KEY"] = "jina_a5fe318b0e0148a7b1b84f9d11bb2911ngUuEpvXV6rUeFVOfTgX8m6B6Qv2"
os.environ['DEEPSEEK_API_KEY'] = "sk-0f87344ec4d74b9ebf4e8dad6b7dbb47"
os.environ["SEARXNG_INSTANCE_URL"] = "https://searx.be/"  # Searxng实例URL
# 用 LiteLLMModel 代替 openai.OpenAI client
model = LiteLLMModel(
    "deepseek/deepseek-chat",
    temperature=0.2,
    provider="deepseek"
)

# 创建OpenDeepSearchTool工具
search_tool = OpenDeepSearchTool(
    model_name="deepseek/deepseek-chat",
    reranker="jina", 
    search_provider="searxng",  # 使用Searxng作为搜索提供者
    # search_provider="serper",
    # serper_api_key=os.environ["SERPER_API_KEY"]
)

# 手动调用setup方法来初始化search_tool
search_tool.setup()

# 创建普通搜索代理用于获取其他信息
agent = OpenDeepSearchAgent(
    model="deepseek/deepseek-chat",
    reranker="jina", 
    search_provider="serper"
)

# 运行查询
query = """
这条消息是真的吗？
时间：2014-03-16
消息内容：
摸奶节是中国云南双柏县鄂家镇彝族传统文化的庆典就是农历的7月14日、15日与16日这三天，包括外来的游人，如果在街上遇见喜欢的女子，都可以摸一摸女子的胸部。姑娘们表面躲躲闪闪，但决无责怪之意因为这是他们这个地区的百姓延续了1000多年的风俗。小伙子以摸到奶为吉祥，姑娘们以被摸而高兴。

要求保留参考信息以及其链接
"""

# 直接使用search方法获取结果
search_response = agent.serp_search.get_sources(query)

# 使用search_tool进行搜索获取最终结果
search_result = search_tool.forward(query)

# 创建工具调用代理进行最终回答
tool_agent = ToolCallingAgent(
    tools=[search_tool],
    model=model,
)
final_result = tool_agent.run(query)

print("搜索结果:")
print(search_result)

print("\n最终回答:")
print(final_result)