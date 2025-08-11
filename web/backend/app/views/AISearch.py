import logging
import time
from flask import Blueprint, request, Response, stream_with_context
from app.utils.HttpResponse import success_response, error_response
from AISearch.utils.search_web import Search
from AISearch.utils.response import search_generate
from AISearch.utils.crawl_web import Crawl
from AISearch.utils.rrf import Retrieval

logger = logging.getLogger(__name__)
search_api = Blueprint('search', __name__)

@search_api.route('/api/search', methods=['POST'])
def perform_search():
    """执行搜索并返回结果的流式API"""
    try:
        query = request.json.get('query')
        if not query:
            return error_response(400, "查询内容不能为空")

        @stream_with_context
        def generate_search_results():
            try:
                start_time = time.time()
                
                # 1. 执行搜索
                yield "正在搜索相关内容...\n"
                search_engine = Search()
                search_result, grade, key_words = search_engine.search(query)
                yield f"搜索关键词：{key_words}\n"
                
                # 2. 爬取网页
                yield "正在获取网页内容...\n"
                crawl = Crawl()
                web_pages = crawl.crawl(search_result)
                yield f"获取到 {len(web_pages)} 条网页内容\n"
                
                # 3. 简化处理：对于简单问题只使用前5个结果
                if grade == 1 and len(web_pages) > 5:
                    web_pages = web_pages[:5]
                    yield "由于问题较为简单，仅使用前5条内容\n"
                
                # 4. 文本召回
                yield "正在分析相关内容...\n"
                queries = [query, key_words]
                retrieval = Retrieval()
                retrieve_results = retrieval.retrieve(web_pages, queries)
                yield f"找到 {len(retrieve_results)} 条相关内容\n"
                
                # 5. 计时
                end_time = time.time()
                yield f"搜索分析用时: {end_time - start_time:.2f} 秒\n"
                
                # 6. 生成回答
                yield "正在生成回答...\n"
                yield from search_generate(query, retrieve_results)
                
            except Exception as e:
                logger.exception(f"搜索流程异常: {str(e)}")
                yield f"\n搜索过程出错: {str(e)}"
        
        return Response(generate_search_results(), content_type='text/plain')
    
    except Exception as e:
        logger.exception(f"搜索API异常: {str(e)}")
        return error_response(500, f"搜索服务异常: {str(e)}")

# 搜索api，返回搜索结果

@search_api.route('/api/search/json', methods=['POST'])
def json_search():
    """JSON版搜索API，返回标准JSON响应而非流式响应"""
    try:
        # 验证请求内容类型
        if not request.is_json:
            return error_response(415, "请求必须是JSON格式，请设置Content-Type: application/json")
        
        query = request.json.get('query')
        if not query:
            return error_response(400, "查询内容不能为空")
        
        # 执行搜索流程
        search_engine = Search()
        search_result, grade, key_words = search_engine.search(query)
        
        # 爬取网页
        crawl = Crawl()
        web_pages = crawl.crawl(search_result)
        
        # 文本召回
        queries = [query, key_words]
        retrieval = Retrieval()
        retrieve_results = retrieval.retrieve(web_pages, queries)
        
        # 获取原始重排序结果(前10条)
        rerank_results = []
        for result in retrieve_results[:10]:
            # 处理 Document 对象
            if hasattr(result, 'metadata'):
                # Document 对象处理
                metadata = result.metadata
                reranked_item = {
                    "title": str(metadata.get('title', '无标题')),
                    "url": str(metadata.get('url', '无URL')),
                    "score": float(metadata.get('score', 0.0)) if 'score' in metadata else 0.0,
                    "snippet": str(result.page_content)[:200] + "..." if len(str(result.page_content)) > 200 else str(result.page_content)
                }
            else:
                # 字典处理 (以防万一)
                reranked_item = {
                    "title": str(result.get("title", "无标题")),
                    "url": str(result.get("url", "无URL")),
                    "score": float(result.get("score", 0.0)) if "score" in result else 0.0,
                    "snippet": str(result.get("snippet", ""))[:200] + "..." if result.get("snippet") else ""
                }
            
            rerank_results.append(reranked_item)
        
        # 将详细内容结果转换为可序列化的格式
        serializable_results = []
        for result in retrieve_results[:5]:  # 仅处理前5条结果
            if hasattr(result, 'metadata'):
                # Document 对象处理
                metadata = result.metadata
                serialized_item = {
                    "content": str(result.page_content)[:500] + "..." if len(str(result.page_content)) > 500 else str(result.page_content),
                    "title": str(metadata.get('title', '无标题')),
                    "url": str(metadata.get('url', '无URL')),
                    "score": float(metadata.get('score', 0.0)) if 'score' in metadata else 0.0
                }
            else:
                # 字典处理 (以防万一)
                serialized_item = {
                    "content": str(result.get("content", ""))[:500] + "..." if len(str(result.get("content", ""))) > 500 else str(result.get("content", "")),
                    "title": str(result.get("title", "")),
                    "url": str(result.get("url", "")),
                    "score": float(result.get("score", 0.0)) if "score" in result else 0.0
                }
                
                # 添加其他可能需要的字段
                if "snippet" in result:
                    serialized_item["snippet"] = str(result["snippet"])
            
            serializable_results.append(serialized_item)
        
        try:
            # 创建生成器
            generator = search_generate(query, retrieve_results)
            
            # 收集生成器的所有内容
            final_answer_parts = []
            for chunk in generator:
                if chunk:  # 忽略空字符串
                    final_answer_parts.append(chunk)
            
            # 合并成一个字符串
            final_answer = "".join(final_answer_parts)
            
            if not final_answer.strip():
                final_answer = "未能生成有效回答，请查看搜索结果。"
        except Exception as e:
            logger.warning(f"生成最终回答时出现错误: {str(e)}")
            final_answer = f"生成回答时遇到错误: {str(e)}"
        
        # 返回完整的可序列化结果
        return success_response({
            "query": query,
            "keywords": key_words,
            "grade": grade,  # 添加问题难度等级
            "sources_count": len(web_pages),
            "rerank_results": rerank_results,  # 添加重排序结果
            "search_results": serializable_results,  # 详细内容结果
            "final_answer": final_answer  # 添加最终生成的回答
        })
    
    except Exception as e:
        logger.exception(f"JSON搜索API异常: {str(e)}")
        return error_response(500, f"搜索服务异常: {str(e)}")