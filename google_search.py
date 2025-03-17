import requests

# API 密钥和 Custom Search Engine ID
API_KEY = "AIzaSyDybHb1hb-zBTcagRf32p70qOON_mdV1Gc"
CSE_ID = "535e3de47f1b0499e"  # 这里需要注意，你的CSE_ID是否正确，用户提供的片段中给出的是"535e3de47f1b0499e"，但你可能需要根据实际情况调整

# 搜索关键词
QUERY = "Python教程"  # 示例关键词，你可以根据需要修改

# Google Custom Search API URL
BASE_URL = "https://www.googleapis.com/customsearch/v1"

# 构建 API 请求参数
params = {
    "key": API_KEY,
    "cx": CSE_ID,
    "q": QUERY,
    "num": 10  # 每次返回结果数量
}

# 发送 API 请求
response = requests.get(BASE_URL, params=params)

# 检查响应状态
if response.status_code == 200:
    # 解析 JSON 数据
    data = response.json()
    # 打印搜索结果
    print("Search Results:")
    for item in data.get("items", []):
        print(f"Title: {item['title']}")
        print(f"Link: {item['link']}")
        print(f"Snippet: {item['snippet']}")
        print("-" * 50)
else:
    print(f"Error: {response.status_code}")
    print(f"Error Message: {response.text}")