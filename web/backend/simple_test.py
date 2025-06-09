from opendeepsearch import OpenDeepSearchTool
import os

# Set environment variables for API keys# Or for SearXNG
# os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

os.environ["SERPER_API_KEY"] = "d2406b655457ab10b1bd94a1469ea1ce42a78984"  # Serper
os.environ["JINA_API_KEY"] = "jina_a5fe318b0e0148a7b1b84f9d11bb2911ngUuEpvXV6rUeFVOfTgX8m6B6Qv2"
os.environ['DEEPSEEK_API_KEY'] = "sk-0f87344ec4d74b9ebf4e8dad6b7dbb47"
os.environ["SEARXNG_INSTANCE_URL"] = "https://searx.be/"  # Searxng实例URL

# Using Serper (default)
# search_agent = OpenDeepSearchTool(
#     model_name="deepseek/deepseek-chat",
#     reranker="jina"
# )

# Or using SearXNG
search_agent = OpenDeepSearchTool(
    model_name="deepseek/deepseek-chat",
    reranker="jina",
    search_provider="searxng",
    searxng_instance_url="https://searx.be/",
    # searxng_api_key="your-api-key-here"  # Optional
)

if not search_agent.is_initialized:
    search_agent.setup()
    
query = "云南省双柏县鄂家镇所谓的“摸奶节”是什么? 给出相关的参考信息和链接。"
result = search_agent.forward(query)
print(result)