from flask import Blueprint, request, jsonify
import requests
import logging
from utils.HttpResponse import success_response, error_response

logger = logging.getLogger(__name__)

system_api = Blueprint('system', __name__)

@system_api.route('/api/system/test-remote-model', methods=['POST'])
def test_remote_model():
    """测试远程模型API连接"""
    try:
        data = request.get_json()
        if not data:
            return error_response(400, "请求数据不能为空")
        
        # 获取参数
        model_type = data.get('model_type')
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        model_name = data.get('model_name')
        
        # 参数验证
        if not all([model_type, api_key, base_url, model_name]):
            return error_response(400, "缺少必要参数")
        
        # 确保base_url格式正确
        if not base_url.endswith('/'):
            base_url += '/'
        
        # 构建完整的API URL
        api_url = f"{base_url}chat/completions"
        
        # 准备请求头
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 准备测试请求体
        test_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a connection test. Please respond with 'OK'."
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        logger.info(f"测试远程模型API: {api_url}")
        logger.info(f"模型类型: {model_type}, 模型名称: {model_name}")
        
        # 发送测试请求
        response = requests.post(
            api_url,
            headers=headers,
            json=test_payload,
            timeout=30  # 30秒超时
        )
        
        # 检查响应状态
        if response.status_code == 200:
            response_data = response.json()
            
            # 检查响应格式是否符合OpenAI标准
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0].get('message', {}).get('content', '')
                
                return success_response(
                    message=f"远程{model_type}模型API连接成功",
                    data={
                        "model_type": model_type,
                        "model_name": model_name,
                        "status": "connected",
                        "response_preview": content[:50] + "..." if len(content) > 50 else content,
                        "usage": response_data.get('usage', {})
                    }
                )
            else:
                return error_response(500, "API响应格式不符合OpenAI标准")
                
        elif response.status_code == 401:
            return error_response(401, "API Key无效或已过期")
        elif response.status_code == 403:
            return error_response(403, "API访问被拒绝，请检查权限")
        elif response.status_code == 404:
            return error_response(404, "API端点不存在，请检查base_url")
        elif response.status_code == 429:
            return error_response(429, "API请求频率限制，请稍后重试")
        else:
            error_text = response.text[:200] if response.text else "未知错误"
            return error_response(
                response.status_code, 
                f"API请求失败: {error_text}"
            )
            
    except requests.exceptions.Timeout:
        logger.error("远程模型API测试超时")
        return error_response(408, "连接超时，请检查网络或API服务是否正常")
        
    except requests.exceptions.ConnectionError:
        logger.error("远程模型API连接错误")
        return error_response(502, "无法连接到API服务，请检查base_url是否正确")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"远程模型API请求异常: {str(e)}")
        return error_response(500, f"请求异常: {str(e)}")
        
    except Exception as e:
        logger.exception("测试远程模型API时发生未知错误")
        return error_response(500, f"测试失败: {str(e)}")

# 添加其他占位API（暂时返回未实现）
@system_api.route('/api/system/test-local-model', methods=['POST'])
def test_local_model():
    """测试本地模型连接 - 占位API"""
    return error_response(501, "本地模型测试功能暂未实现")

@system_api.route('/api/system/test-platform-cookie', methods=['POST'])
def test_platform_cookie():
    """测试平台Cookie - 占位API"""
    return error_response(501, "平台Cookie测试功能暂未实现")

@system_api.route('/api/system/test-crawler', methods=['POST'])
def test_crawler():
    """测试爬虫连接 - 占位API"""
    return error_response(501, "爬虫连接测试功能暂未实现")

@system_api.route('/api/system/settings', methods=['GET'])
def get_settings():
    """获取系统设置 - 占位API"""
    return error_response(501, "获取系统设置功能暂未实现")

@system_api.route('/api/system/settings', methods=['POST'])
def save_settings():
    """保存系统设置 - 占位API"""
    return error_response(501, "保存系统设置功能暂未实现")

@system_api.route('/api/system/platform-cookies', methods=['GET'])
def get_platform_cookies():
    """获取平台Cookies - 占位API"""
    return error_response(501, "获取平台Cookies功能暂未实现")

@system_api.route('/api/system/crawler-settings', methods=['POST'])
def save_crawler_settings():
    """保存爬虫设置 - 占位API"""
    return error_response(501, "保存爬虫设置功能暂未实现")

@system_api.route('/api/system/clear-cache', methods=['POST'])
def clear_cache():
    """清除缓存 - 占位API"""
    return error_response(501, "清除缓存功能暂未实现")

@system_api.route('/api/system/restart-service', methods=['POST'])
def restart_service():
    """重启服务 - 占位API"""
    return error_response(501, "重启服务功能暂未实现")