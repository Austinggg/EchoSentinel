import requests
import logging
from typing import Dict, Any, Optional
from smolagents.tools import Tool
import datetime  # 添加这一行导入

class AISearchTool(Tool):
    """使用自定义 AISearch API 的搜索工具，符合 smolagents.Tool 接口"""
    
    name = "search"
    description = "搜索网络信息以查找相关内容。适用于需要最新信息或特定事实的查询。"
    
    # inputs 定义了工具接受的输入参数
    inputs = {
        "query": {
            "type": "string",
            "description": "搜索查询字符串"
        }
    }
    
    # 添加 output_type 属性
    output_type = "string"
    
    def __init__(self, api_url="http://127.0.0.1:8000"):
        """初始化 AISearch 工具
        
        Args:
            api_url: AISearch API 的基础URL
        """
        super().__init__()
        self.api_url = api_url
        self.logger = logging.getLogger(__name__)
        self.last_search_data = {}  # 存储最后一次搜索的原始数据
        
    # 修改 forward 方法中保存搜索结果的部分
    
    def forward(self, query: str) -> str:
        """实现 Tool 基类所需的 forward 方法"""
        if not query:
            return "错误：没有提供搜索查询"
            
        try:
            # 记录开始时间
            start_time = datetime.datetime.now()
            
            # 调用 JSON 版本的 API
            response = requests.post(
                f"{self.api_url}/api/search/json",
                headers={"Content-Type": "application/json"},
                json={"query": query},
                timeout=240
            )
            
            # 记录结束时间
            end_time = datetime.datetime.now()
            
            # 检查是否成功
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == 200:
                search_results = data["data"]
                
                # 保存搜索结果，确保是字典类型
                if isinstance(search_results, dict):
                    # 添加搜索性能统计
                    search_results["search_performance"] = {
                        "duration": (end_time - start_time).total_seconds(),
                        "timestamp": end_time.isoformat(),
                        "query_size": len(query)
                    }
                    
                    self.last_search_data = search_results
                else:
                    self.last_search_data = {"error": "搜索结果格式异常"}
                    self.logger.warning(f"搜索结果不是字典格式: {type(search_results)}")
                
                return self._format_results(search_results if isinstance(search_results, dict) else {}, query)
            else:
                self.logger.error(f"AISearch API 返回错误: {data.get('message')}")
                # 重置搜索结果
                self.last_search_data = {"error": data.get('message', '未知错误')}
                return f"搜索错误: {data.get('message', '未知错误')}"
                    
        except requests.RequestException as e:
            self.logger.error(f"调用 AISearch API 失败: {str(e)}")
            # 重置搜索结果
            self.last_search_data = {"error": str(e)}
            return f"搜索服务不可用: {str(e)}"
    
    def _format_results(self, search_results: Dict[str, Any], query: str) -> str:
        """格式化搜索结果为文本，限制大小避免模型输入过大"""
        # 确保 search_results 是字典
        if not isinstance(search_results, dict):
            self.logger.error(f"_format_results 收到非字典类型: {type(search_results)}")
            return f"搜索结果格式错误。查询: {query}"
        
        # 提取主要信息并格式化为文本结果
        formatted_results = [
            f"# 搜索结果: {search_results.get('query', query)}\n",
            f"关键词: {search_results.get('keywords', '')}\n",
            f"找到 {search_results.get('sources_count', 0)} 个相关来源\n\n"
        ]
        
        # 添加重排序后的结果，但限制数量并精简内容
        formatted_results.append("## 相关内容摘要:\n")
        rerank_results = search_results.get("rerank_results", [])
        if not isinstance(rerank_results, list):
            self.logger.error(f"rerank_results 不是列表: {type(rerank_results)}")
            rerank_results = []
        
        # 仅包含前3个最相关结果
        for i, result in enumerate(rerank_results[:3], 1):
            if not isinstance(result, dict):
                continue
                
            # 限制摘要长度
            snippet = result.get('snippet', '无摘要')
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."
                
            formatted_results.append(f"{i}. 【{result.get('title', '无标题')[:50]}】\n")
            formatted_results.append(f"   {snippet}\n")
            formatted_results.append(f"   来源: {result.get('url', '无链接')[:80]}\n\n")
        
        # 只添加最重要的1-2个详细内容
        search_results_data = search_results.get("search_results")
        if isinstance(search_results_data, list) and len(search_results_data) > 0:
            formatted_results.append("## 详细内容:\n")
            for i, result in enumerate(search_results_data[:2], 1):
                if not isinstance(result, dict):
                    continue
                    
                # 限制内容长度
                content = result.get('content', '无内容')
                if len(content) > 300:  # 大幅减少详细内容的长度
                    content = content[:300] + "..."
                    
                formatted_results.append(f"### {i}. {result.get('title', '无标题')[:50]}\n")
                formatted_results.append(f"{content}\n\n")
        
        result_text = "".join(formatted_results)
        
        # 最终确保总长度不超过5000字符
        if len(result_text) > 5000:
            return result_text[:4950] + "\n...(内容已截断)"
        
        return result_text