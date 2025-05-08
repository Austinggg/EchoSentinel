import re
import uuid
import os
from datetime import datetime
import urllib.parse
from pathlib import Path
import threading
from flask import Blueprint, request, Response
import requests
import logging
# 导入视频处理相关函数
from api.videoUpload import UPLOAD_DIR, THUMBNAIL_DIR, generate_video_thumbnail, auto_process_video
from utils.database import VideoFile, db
from utils.HttpResponse import HttpResponse
import json
# 定义下载目录
BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent.parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 创建Blueprint
douyin_api = Blueprint('douyin_api', __name__)

# 配置Docker服务地址
DOCKER_SERVICE_HOST = "http://localhost:80"

# 设置日志
logger = logging.getLogger(__name__)
@douyin_api.route('/api/download', methods=['GET'])
def download_social_media_video():
    """在线下载抖音/TikTok视频API"""
    try:
        # 获取参数
        video_url = request.args.get('url')
        prefix = request.args.get('prefix', 'true').lower() == 'true'
        with_watermark = request.args.get('with_watermark', 'false').lower() == 'true'
        
        if not video_url:
            return {"code": 400, "message": "缺少URL参数"}, 400
            
        # 解码URL
        video_url = urllib.parse.unquote(video_url)
        logger.info(f"收到下载请求，URL: {video_url}")
        
        # 判断平台类型
        platform = "douyin" if "douyin.com" in video_url else "tiktok" if "tiktok.com" in video_url else "unknown"
        
        if platform == "unknown":
            return {"code": 400, "message": "不支持的平台，仅支持抖音和TikTok"}, 400
            
        # 为下载的视频创建专门的目录
        platform_dir = DOWNLOAD_DIR / platform
        platform_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        
        # 调用Docker服务下载视频
        api_endpoint = "/api/download" 
        
        response = requests.get(
            f"{DOCKER_SERVICE_HOST}{api_endpoint}",
            params={
                "url": video_url,
                "prefix": str(prefix).lower(),
                "with_watermark": str(with_watermark).lower()
            }
        )
        
        if response.status_code != 200:
            return {"code": 500, "message": f"下载服务请求失败: HTTP {response.status_code}"}, 500
            
        data = response.json()
        if data.get("code") != 200:
            return {"code": 500, "message": f"下载失败: {data.get('message')}"}, 500
            
        # 获取视频信息
        video_info = data.get("data", {})
        video_url = video_info.get("video_url")
        
        if not video_url:
            return {"code": 404, "message": "没有找到可下载的视频URL"}, 404
            
        # 从返回数据中提取元数据
        aweme_id = video_info.get("aweme_id", "")
        desc = video_info.get("desc", "无标题")
        create_time_str = video_info.get("create_time", "")
        
        # 获取视频作者信息
        author = video_info.get("author", {})
        nickname = author.get("nickname", "未知作者")
        
        # 解析创建时间
        create_time = None
        if create_time_str and create_time_str.isdigit():
            create_time = datetime.fromtimestamp(int(create_time_str))
        else:
            create_time = datetime.now()
            
        # 提取标签
        hashtags = video_info.get("hashtags", [])
        tags = ["#" + tag.get("hashtag_name") for tag in hashtags if tag.get("hashtag_name")]
        
        # 处理文件名（以时间+标题+标签方式命名，与解析器兼容）
        time_prefix = create_time.strftime('%Y-%m-%d %H-%M-%S')
        
        # 构建完整文件名
        file_components = [time_prefix]
        if desc and desc != "无标题":
            # 清理描述中的特殊字符
            clean_desc = re.sub(r'[\\/*?:"<>|]', "", desc)
            clean_desc = clean_desc.replace(" ", "_").replace("\n", "_")
            if len(clean_desc) > 50:  # 限制标题长度
                clean_desc = clean_desc[:47] + "..."
            file_components.append(clean_desc)
        
        if tags:
            file_components.extend(tags)
            
        file_components.append("video")
        formatted_filename = "_".join(file_components)
        
        # 下载视频文件
        video_response = requests.get(video_url, stream=True)
        if video_response.status_code != 200:
            return {"code": 500, "message": f"下载视频失败: HTTP {video_response.status_code}"}, 500
            
        # 确定文件扩展名
        content_type = video_response.headers.get('content-type', '')
        if 'video/mp4' in content_type:
            file_ext = 'mp4'
        elif 'video/webm' in content_type:
            file_ext = 'webm'
        else:
            # 从URL尝试提取扩展名
            url_path = urllib.parse.urlparse(video_url).path
            if '.' in url_path:
                file_ext = url_path.split('.')[-1].lower()
                if len(file_ext) > 5:  # 确保扩展名合理
                    file_ext = 'mp4'  # 默认扩展名
            else:
                file_ext = 'mp4'  # 默认扩展名
                
        # 完整的文件路径
        filename_with_ext = f"{formatted_filename}.{file_ext}"
        download_path = platform_dir / filename_with_ext
        upload_path = UPLOAD_DIR / f"{file_id}.{file_ext}"
        
        # 保存视频文件
        with open(download_path, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # 复制一份到上传目录（使用文件ID）
        import shutil
        shutil.copy2(str(download_path), str(upload_path))
        
        # 获取文件大小
        file_size = os.path.getsize(upload_path)
        
        # 保存到数据库
        new_video = VideoFile(
            id=file_id,
            filename=desc,
            extension=file_ext,
            size=file_size,
            mime_type=f"video/{file_ext}",
            user_id=None,
            upload_time=datetime.utcnow(),
            publish_time=create_time,
            tags=','.join(tags),
            status="processing",
            source_url=video_url,
            source_platform=platform,
            source_id=aweme_id
        )
        
        db.session.add(new_video)
        db.session.commit()
        
        logger.info(f"视频下载成功: {download_path}, ID: {file_id}")
        
        # 启动缩略图生成和处理
        try:
            # 生成缩略图（在后台执行，不阻塞响应）
            thumbnail_thread = threading.Thread(
                target=generate_video_thumbnail,
                args=(str(upload_path), file_id)
            )
            thumbnail_thread.daemon = True
            thumbnail_thread.start()
            
            # 启动自动处理流程
            processing_thread = threading.Thread(
                target=auto_process_video,
                args=(file_id,)
            )
            processing_thread.daemon = True
            processing_thread.start()
        except Exception as e:
            logger.error(f"启动后处理任务失败: {str(e)}")
        
        # 返回成功响应
        return {
            "code": 200, 
            "message": "下载成功",
            "data": {
                "fileId": file_id,
                "filename": desc,
                "size": file_size,
                "mimeType": f"video/{file_ext}",
                "url": f"/api/videos/{file_id}",
                "thumbnail": f"/api/videos/{file_id}/thumbnail",
                "publishTime": create_time.isoformat() if create_time else None,
                "tags": tags,
                "platform": platform,
                "originalUrl": video_url,
                "downloadPath": str(download_path)
            }
        }
        
    except Exception as e:
        logger.exception(f"下载视频失败: {str(e)}")
        db.session.rollback()
        return {"code": 500, "message": f"下载失败: {str(e)}"}, 500
    
@douyin_api.route('/api/douyin/web/handler_user_profile', methods=['GET'])
def proxy_douyin_user_profile():
    """
    代理抖音用户资料请求到Docker服务
    """
    try:
        # 获取原始请求的查询参数
        query_params = request.args.to_dict()
        
        # 构建目标URL
        target_url = f"{DOCKER_SERVICE_HOST}/api/douyin/web/handler_user_profile"
        
        # 发送请求到Docker服务
        response = requests.get(
            target_url,
            params=query_params,
            headers={k: v for k, v in request.headers if k.lower() not in ['host', 'content-length']}
        )
        
        # 记录请求信息
        logger.info(f"Proxied request to: {target_url} with params: {query_params}")
        logger.debug(f"Response status: {response.status_code}")
        
        # 准备要返回的数据
        content = response.content
        status_code = response.status_code
        headers = {
            key: value for key, value in response.headers.items()
            if key.lower() not in ['content-length', 'transfer-encoding', 'connection']
        }
        
        # 返回代理的响应
        return Response(
            content, 
            status=status_code,
            headers=headers,
            content_type=response.headers.get('content-type', 'application/json')
        )
    
    except Exception as e:
        logger.exception(f"Error proxying request to douyin API: {str(e)}")
        return {"code": 500, "message": f"代理请求失败: {str(e)}"}, 500

@douyin_api.route('/api/douyin/web/<path:path>', methods=['GET', 'POST'])
def proxy_other_douyin_endpoints(path):
    """
    通用代理处理程序，可以代理其他抖音API端点
    """
    try:
        # 获取原始请求的查询参数
        query_params = request.args.to_dict()
        
        # 构建目标URL
        target_url = f"{DOCKER_SERVICE_HOST}/api/douyin/web/{path}"
        
        # 根据请求方法处理
        if request.method == 'GET':
            response = requests.get(
                target_url,
                params=query_params,
                headers={k: v for k, v in request.headers if k.lower() not in ['host', 'content-length']}
            )
        else:  # POST
            response = requests.post(
                target_url,
                params=query_params,
                json=request.get_json(silent=True),
                data=request.form if not request.is_json else None,
                headers={k: v for k, v in request.headers if k.lower() not in ['host', 'content-length']}
            )
        
        # 记录请求信息
        logger.info(f"Proxied {request.method} request to: {target_url}")
        
        # 返回代理的响应
        return Response(
            response.content, 
            status=response.status_code,
            headers={
                key: value for key, value in response.headers.items()
                if key.lower() not in ['content-length', 'transfer-encoding', 'connection']
            },
            content_type=response.headers.get('content-type', 'application/json')
        )
    
    except Exception as e:
        logger.exception(f"Error proxying request to douyin API path {path}: {str(e)}")
        return {"code": 500, "message": f"代理请求失败: {str(e)}"}, 500
# 在现有代码中添加这个函数
@douyin_api.route('/api/douyin/web/fetch_user_post_videos', methods=['GET'])
def proxy_douyin_user_videos():
    """代理抖音用户视频列表请求"""
    try:
        # 获取原始请求的查询参数
        query_params = request.args.to_dict()
        
        # 构建目标URL
        target_url = f"{DOCKER_SERVICE_HOST}/api/douyin/web/fetch_user_post_videos"
        
        # 发送请求到Docker服务
        response = requests.get(
            target_url,
            params=query_params,
            headers={k: v for k, v in request.headers if k.lower() not in ['host', 'content-length']}
        )
        
        # 记录请求信息
        logger.info(f"Proxied request to: {target_url} with params: {query_params}")
        
        # 返回代理的响应
        return Response(
            response.content, 
            status=response.status_code,
            headers={
                key: value for key, value in response.headers.items()
                if key.lower() not in ['content-length', 'transfer-encoding', 'connection']
            },
            content_type=response.headers.get('content-type', 'application/json')
        )
    
    except Exception as e:
        logger.exception(f"Error proxying request to douyin video list API: {str(e)}")
        return {"code": 500, "message": f"代理请求失败: {str(e)}"}, 500



