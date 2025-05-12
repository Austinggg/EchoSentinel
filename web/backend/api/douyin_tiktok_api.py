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
from utils.database import VideoFile, db, DouyinVideo  # 添加DouyinVideo导入
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
"""在线下载抖音/TikTok视频，并且存入数据库开启分析"""
@douyin_api.route('/api/download_and_analyze', methods=['GET'])
def download_and_analyze_video():
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
        
        # 尝试从URL中提取aweme_id
        aweme_id = None
        if "video/" in video_url:
            # 格式如: https://www.douyin.com/video/7372484719365098803
            match = re.search(r'video/(\d+)', video_url)
            if match:
                aweme_id = match.group(1)
                logger.info(f"从URL提取的aweme_id: {aweme_id}")
        
        # 如果找到aweme_id，使用fetch_one_video接口获取详情
        if aweme_id:
            detail_api_endpoint = f"/api/douyin/web/fetch_one_video?aweme_id={aweme_id}"
            logger.info(f"获取视频详情: {DOCKER_SERVICE_HOST}{detail_api_endpoint}")
            
            detail_response = requests.get(f"{DOCKER_SERVICE_HOST}{detail_api_endpoint}")
            if detail_response.status_code != 200:
                logger.error(f"获取视频详情失败: HTTP {detail_response.status_code}")
                return {"code": 500, "message": f"获取视频详情失败: HTTP {detail_response.status_code}"}, 500
            
            try:
                detail_data = detail_response.json()
                if detail_data.get("code") != 200:
                    logger.error(f"获取视频详情失败: {detail_data.get('message')}")
                    return {"code": 500, "message": f"获取视频详情失败: {detail_data.get('message')}"}, 500
                
                # 提取视频详细信息
                aweme_detail = detail_data.get("data", {}).get("aweme_detail", {})
                
                if not aweme_detail:
                    logger.error("返回的数据中缺少aweme_detail")
                    return handle_direct_download(video_url, platform, file_id, prefix, with_watermark)
                
                # 从详情中提取所需数据
                desc = aweme_detail.get("desc", "无标题")
                create_time_str = str(aweme_detail.get("create_time", ""))
                author = aweme_detail.get("author", {})
                nickname = author.get("nickname", "未知作者")
                
                # 提取标签信息
                text_extra = aweme_detail.get("text_extra", [])
                tags = []
                for item in text_extra:
                    if item.get("type") == 1 and item.get("hashtag_name"):
                        tags.append("#" + item.get("hashtag_name"))
                
                # 调用下载API获取视频文件
                download_api_endpoint = "/api/download"
                
                logger.info(f"开始下载视频: {DOCKER_SERVICE_HOST}{download_api_endpoint}")
                
                # 直接下载视频
                download_response = requests.get(
                    f"{DOCKER_SERVICE_HOST}{download_api_endpoint}",
                    params={
                        "url": video_url,
                        "prefix": str(prefix).lower(),
                        "with_watermark": str(with_watermark).lower()
                    },
                    stream=True  # 使用流模式
                )
                
                if download_response.status_code != 200:
                    return {"code": 500, "message": f"下载视频失败: HTTP {download_response.status_code}"}, 500
                
                # 解析创建时间
                create_time = None
                if create_time_str and create_time_str.isdigit():
                    create_time = datetime.fromtimestamp(int(create_time_str))
                else:
                    create_time = datetime.now()
                    
                # 处理文件名
                time_prefix = create_time.strftime('%Y-%m-%d %H-%M-%S')
                
                # 构建完整文件名
                file_components = [time_prefix]
                if desc and desc != "无标题":
                    clean_desc = re.sub(r'[\\/*?:"<>|]', "", desc)
                    clean_desc = clean_desc.replace(" ", "_").replace("\n", "_")
                    if len(clean_desc) > 50:
                        clean_desc = clean_desc[:47] + "..."
                    file_components.append(clean_desc)
                
                if tags:
                    # 为避免文件名过长，只使用前两个标签
                    tag_str = "_".join(tags[:2])
                    file_components.append(tag_str)
                    
                file_components.append("video")
                formatted_filename = "_".join(file_components)
                
                # 确定文件扩展名
                content_type = download_response.headers.get('content-type', '')
                if 'video/mp4' in content_type:
                    file_ext = 'mp4'
                elif 'video/webm' in content_type:
                    file_ext = 'webm'
                else:
                    file_ext = 'mp4'  # 默认扩展名
                
                # 完整的文件路径
                filename_with_ext = f"{formatted_filename}.{file_ext}"
                download_path = platform_dir / filename_with_ext
                upload_path = UPLOAD_DIR / f"{file_id}.{file_ext}"
                
                # 保存视频文件
                with open(download_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 复制一份到上传目录
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

                # 添加以下代码：检查并更新DouyinVideo与VideoFile的关联
                if aweme_id:
                    existing_douyin_video = DouyinVideo.query.filter_by(aweme_id=aweme_id).first()
                    if existing_douyin_video:
                        # 更新关联
                        existing_douyin_video.video_file_id = file_id
                        db.session.commit()
                        logger.info(f"更新视频关联: DouyinVideo (aweme_id: {aweme_id}) -> VideoFile (id: {file_id})")
                    else:
                        logger.info(f"未找到抖音视频记录: aweme_id={aweme_id}")

                logger.info(f"视频下载成功: {download_path}, ID: {file_id}")
                
                # 启动后台处理
                try:
                    # 启动缩略图生成
                    thumbnail_thread = threading.Thread(
                        target=generate_video_thumbnail,
                        args=(str(upload_path), file_id)
                    )
                    thumbnail_thread.daemon = True
                    thumbnail_thread.start()
                    
                    # 启动视频处理
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
                
            except json.JSONDecodeError:
                logger.warning("视频详情API未返回JSON数据，转为直接下载模式")
                return handle_direct_download(video_url, platform, file_id, prefix, with_watermark)
        else:
            # 如果未能从URL中提取aweme_id，使用直接下载模式
            logger.warning(f"无法从URL中提取aweme_id: {video_url}")
            return handle_direct_download(video_url, platform, file_id, prefix, with_watermark)
            
    except Exception as e:
        logger.exception(f"下载视频失败: {str(e)}")
        db.session.rollback()
        return {"code": 500, "message": f"下载失败: {str(e)}"}, 500
# 添加这个新的API端点，仅下载不分析

@douyin_api.route('/api/download/', methods=['GET'])
def download_video_only():
    """只下载抖音/TikTok视频，不启动分析流程"""
    try:
        # 获取参数
        video_url = request.args.get('url')
        prefix = request.args.get('prefix', 'true').lower() == 'true'
        with_watermark = request.args.get('with_watermark', 'false').lower() == 'true'
        
        if not video_url:
            return {"code": 400, "message": "缺少URL参数"}, 400
            
        # 解码URL
        video_url = urllib.parse.unquote(video_url)
        logger.info(f"收到下载请求(仅下载)，URL: {video_url}")
        
        # 判断平台类型
        platform = "douyin" if "douyin.com" in video_url else "tiktok" if "tiktok.com" in video_url else "unknown"
        
        if platform == "unknown":
            return {"code": 400, "message": "不支持的平台，仅支持抖音和TikTok"}, 400
            
        # 为下载的视频创建专门的目录
        platform_dir = DOWNLOAD_DIR / platform
        platform_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        
        # 尝试从URL中提取aweme_id
        aweme_id = None
        if "video/" in video_url:
            match = re.search(r'video/(\d+)', video_url)
            if match:
                aweme_id = match.group(1)
                logger.info(f"从URL提取的aweme_id: {aweme_id}")
        
        # 如果找到aweme_id，使用fetch_one_video接口获取详情
        if aweme_id:
            detail_api_endpoint = f"/api/douyin/web/fetch_one_video?aweme_id={aweme_id}"
            logger.info(f"获取视频详情: {DOCKER_SERVICE_HOST}{detail_api_endpoint}")
            
            detail_response = requests.get(f"{DOCKER_SERVICE_HOST}{detail_api_endpoint}")
            if detail_response.status_code != 200:
                logger.error(f"获取视频详情失败: HTTP {detail_response.status_code}")
                return handle_direct_download_only(video_url, platform, file_id, prefix, with_watermark)
            
            try:
                detail_data = detail_response.json()
                if detail_data.get("code") != 200:
                    logger.error(f"获取视频详情失败: {detail_data.get('message')}")
                    return handle_direct_download_only(video_url, platform, file_id, prefix, with_watermark)
                
                # 提取视频详细信息
                aweme_detail = detail_data.get("data", {}).get("aweme_detail", {})
                
                if not aweme_detail:
                    logger.error("返回的数据中缺少aweme_detail")
                    return handle_direct_download_only(video_url, platform, file_id, prefix, with_watermark)
                
                # 从详情中提取所需数据
                desc = aweme_detail.get("desc", "无标题")
                create_time_str = str(aweme_detail.get("create_time", ""))
                author = aweme_detail.get("author", {})
                nickname = author.get("nickname", "未知作者")
                
                # 提取标签信息
                text_extra = aweme_detail.get("text_extra", [])
                tags = []
                for item in text_extra:
                    if item.get("type") == 1 and item.get("hashtag_name"):
                        tags.append("#" + item.get("hashtag_name"))
                
                # 调用下载API获取视频文件
                download_api_endpoint = "/api/download"
                logger.info(f"开始下载视频(仅下载): {DOCKER_SERVICE_HOST}{download_api_endpoint}")
                
                # 直接下载视频
                download_response = requests.get(
                    f"{DOCKER_SERVICE_HOST}{download_api_endpoint}",
                    params={
                        "url": video_url,
                        "prefix": str(prefix).lower(),
                        "with_watermark": str(with_watermark).lower()
                    },
                    stream=True  # 使用流模式
                )
                
                if download_response.status_code != 200:
                    return {"code": 500, "message": f"下载视频失败: HTTP {download_response.status_code}"}, 500
                
                # 解析创建时间
                create_time = None
                if create_time_str and create_time_str.isdigit():
                    create_time = datetime.fromtimestamp(int(create_time_str))
                else:
                    create_time = datetime.now()
                    
                # 处理文件名
                time_prefix = create_time.strftime('%Y-%m-%d %H-%M-%S')
                
                # 构建完整文件名
                file_components = [time_prefix]
                if desc and desc != "无标题":
                    clean_desc = re.sub(r'[\\/*?:"<>|]', "", desc)
                    clean_desc = clean_desc.replace(" ", "_").replace("\n", "_")
                    if len(clean_desc) > 50:
                        clean_desc = clean_desc[:47] + "..."
                    file_components.append(clean_desc)
                
                if tags:
                    # 为避免文件名过长，只使用前两个标签
                    tag_str = "_".join(tags[:2])
                    file_components.append(tag_str)
                    
                file_components.append("video")
                formatted_filename = "_".join(file_components)
                
                # 确定文件扩展名
                content_type = download_response.headers.get('content-type', '')
                if 'video/mp4' in content_type:
                    file_ext = 'mp4'
                elif 'video/webm' in content_type:
                    file_ext = 'webm'
                else:
                    file_ext = 'mp4'  # 默认扩展名
                
                # 完整的文件路径
                filename_with_ext = f"{formatted_filename}.{file_ext}"
                download_path = platform_dir / filename_with_ext
                upload_path = UPLOAD_DIR / f"{file_id}.{file_ext}"
                
                # 保存视频文件
                with open(download_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 复制一份到上传目录
                import shutil
                shutil.copy2(str(download_path), str(upload_path))
                
                # 获取文件大小
                file_size = os.path.getsize(upload_path)
                
                # 保存到数据库 - 注意状态为"downloaded"而不是"processing"
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
                    status="downloaded",  # 修改为downloaded状态
                    source_url=video_url,
                    source_platform=platform,
                    source_id=aweme_id
                )

                db.session.add(new_video)
                db.session.commit()

                # 添加以下代码：检查并更新DouyinVideo与VideoFile的关联
                if aweme_id:
                    existing_douyin_video = DouyinVideo.query.filter_by(aweme_id=aweme_id).first()
                    if existing_douyin_video:
                        # 更新关联
                        existing_douyin_video.video_file_id = file_id
                        db.session.commit()
                        logger.info(f"更新视频关联: DouyinVideo (aweme_id: {aweme_id}) -> VideoFile (id: {file_id})")
                    else:
                        logger.info(f"未找到抖音视频记录: aweme_id={aweme_id}")

                logger.info(f"视频下载成功(仅下载): {download_path}, ID: {file_id}")
                
                # 启动缩略图生成，但不启动视频分析
                try:
                    # 只启动缩略图生成
                    thumbnail_thread = threading.Thread(
                        target=generate_video_thumbnail,
                        args=(str(upload_path), file_id)
                    )
                    thumbnail_thread.daemon = True
                    thumbnail_thread.start()
                except Exception as e:
                    logger.error(f"启动缩略图生成失败: {str(e)}")
                
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
                        "downloadPath": str(download_path),
                        "status": "downloaded"  # 明确标明状态是downloaded
                    }
                }
                
            except json.JSONDecodeError:
                logger.warning("视频详情API未返回JSON数据，转为直接下载模式")
                return handle_direct_download_only(video_url, platform, file_id, prefix, with_watermark)
        else:
            # 如果未能从URL中提取aweme_id，使用直接下载模式
            logger.warning(f"无法从URL中提取aweme_id: {video_url}")
            return handle_direct_download_only(video_url, platform, file_id, prefix, with_watermark)
            
    except Exception as e:
        logger.exception(f"下载视频失败: {str(e)}")
        db.session.rollback()
        return {"code": 500, "message": f"下载失败: {str(e)}"}, 500

# 添加用于仅下载的直接下载处理函数
def handle_direct_download_only(video_url, platform, file_id, prefix, with_watermark):
    """直接下载处理（无元数据），不启动分析"""
    try:
        platform_dir = DOWNLOAD_DIR / platform
        
        # 直接下载视频文件
        download_response = requests.get(
            f"{DOCKER_SERVICE_HOST}/api/download",
            params={
                "url": video_url,
                "prefix": str(prefix).lower(),
                "with_watermark": str(with_watermark).lower()
            },
            stream=True
        )
        
        if download_response.status_code != 200:
            return {"code": 500, "message": f"下载视频失败: HTTP {download_response.status_code}"}, 500
        
        # 使用当前时间作为文件名
        time_prefix = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        formatted_filename = f"{time_prefix}_video"
        
        # 确定文件扩展名
        content_type = download_response.headers.get('content-type', '')
        if 'video/mp4' in content_type:
            file_ext = 'mp4'
        elif 'video/webm' in content_type:
            file_ext = 'webm'
        else:
            file_ext = 'mp4'  # 默认扩展名
            
        # 完整的文件路径
        filename_with_ext = f"{formatted_filename}.{file_ext}"
        download_path = platform_dir / filename_with_ext
        upload_path = UPLOAD_DIR / f"{file_id}.{file_ext}"
        
        # 保存视频文件
        with open(download_path, 'wb') as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        # 复制一份到上传目录
        import shutil
        shutil.copy2(str(download_path), str(upload_path))
        
        # 获取文件大小
        file_size = os.path.getsize(upload_path)
        
        # 保存到数据库（简化信息）- 修改status为downloaded
        new_video = VideoFile(
            id=file_id,
            filename=f"从{platform}下载的视频",
            extension=file_ext,
            size=file_size,
            mime_type=f"video/{file_ext}",
            user_id=None,
            upload_time=datetime.utcnow(),
            publish_time=datetime.utcnow(),
            status="downloaded",  # 修改为downloaded状态
            source_url=video_url,
            source_platform=platform
        )
    
        db.session.add(new_video)
        db.session.commit()

        # 尝试从URL提取aweme_id并更新关联
        extracted_aweme_id = None
        if "video/" in video_url:
            match = re.search(r'video/(\d+)', video_url)
            if match:
                extracted_aweme_id = match.group(1)
                
        if extracted_aweme_id:
            existing_douyin_video = DouyinVideo.query.filter_by(aweme_id=extracted_aweme_id).first()
            if existing_douyin_video:
                # 更新关联
                existing_douyin_video.video_file_id = file_id
                db.session.commit()
                logger.info(f"更新视频关联(直接下载): DouyinVideo (aweme_id: {extracted_aweme_id}) -> VideoFile (id: {file_id})")

        logger.info(f"视频下载成功（仅下载）: {download_path}, ID: {file_id}")
                
        # 只启动缩略图生成，不启动视频处理
        try:
            # 启动缩略图生成
            thumbnail_thread = threading.Thread(
                target=generate_video_thumbnail,
                args=(str(upload_path), file_id)
            )
            thumbnail_thread.daemon = True
            thumbnail_thread.start()
            
            # 不再启动视频处理
            # processing_thread = threading.Thread(
            #     target=auto_process_video,
            #     args=(file_id,)
            # )
            # processing_thread.daemon = True
            # processing_thread.start()
            
        except Exception as e:
            logger.error(f"启动缩略图生成失败: {str(e)}")
        
        return {
            "code": 200, 
            "message": "下载成功",
            "data": {
                "fileId": file_id,
                "filename": f"从{platform}下载的视频",
                "size": file_size,
                "mimeType": f"video/{file_ext}",
                "url": f"/api/videos/{file_id}",
                "thumbnail": f"/api/videos/{file_id}/thumbnail",
                "platform": platform,
                "originalUrl": video_url,
                "downloadPath": str(download_path),
                "status": "downloaded"  # 明确标明状态是downloaded
            }
        }
    
    except Exception as e:
        logger.exception(f"直接下载处理失败: {str(e)}")
        return {"code": 500, "message": f"下载失败: {str(e)}"}, 500
# 抖音单个视频详情
@douyin_api.route('/api/douyin/web/fetch_one_video', methods=['GET'])
def proxy_douyin_one_video():
    """代理抖音单个视频详情请求"""
    try:
        # 获取原始请求的查询参数
        query_params = request.args.to_dict()
        
        # 构建目标URL
        target_url = f"{DOCKER_SERVICE_HOST}/api/douyin/web/fetch_one_video"
        
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
        logger.exception(f"Error proxying request to douyin video detail API: {str(e)}")
        return {"code": 500, "message": f"代理请求失败: {str(e)}"}, 500
def handle_direct_download(video_url, platform, file_id, prefix, with_watermark):
    """直接下载处理（无元数据）"""
    try:
        platform_dir = DOWNLOAD_DIR / platform
        
        # 直接下载视频文件
        download_response = requests.get(
            f"{DOCKER_SERVICE_HOST}/api/download",
            params={
                "url": video_url,
                "prefix": str(prefix).lower(),
                "with_watermark": str(with_watermark).lower()
            },
            stream=True
        )
        
        if download_response.status_code != 200:
            return {"code": 500, "message": f"下载视频失败: HTTP {download_response.status_code}"}, 500
        
        # 使用当前时间作为文件名
        time_prefix = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        formatted_filename = f"{time_prefix}_video"
        
        # 确定文件扩展名
        content_type = download_response.headers.get('content-type', '')
        if 'video/mp4' in content_type:
            file_ext = 'mp4'
        elif 'video/webm' in content_type:
            file_ext = 'webm'
        else:
            file_ext = 'mp4'  # 默认扩展名
            
        # 完整的文件路径
        filename_with_ext = f"{formatted_filename}.{file_ext}"
        download_path = platform_dir / filename_with_ext
        upload_path = UPLOAD_DIR / f"{file_id}.{file_ext}"
        
        # 保存视频文件
        with open(download_path, 'wb') as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        # 复制一份到上传目录
        import shutil
        shutil.copy2(str(download_path), str(upload_path))
        
        # 获取文件大小
        file_size = os.path.getsize(upload_path)
        
        # 保存到数据库（简化信息）
        new_video = VideoFile(
            id=file_id,
            filename=f"从{platform}下载的视频",
            extension=file_ext,
            size=file_size,
            mime_type=f"video/{file_ext}",
            user_id=None,
            upload_time=datetime.utcnow(),
            publish_time=datetime.utcnow(),
            status="processing",
            source_url=video_url,
            source_platform=platform
        )
    
        db.session.add(new_video)
        db.session.commit()

        # 尝试从URL提取aweme_id并更新关联
        extracted_aweme_id = None
        if "video/" in video_url:
            match = re.search(r'video/(\d+)', video_url)
            if match:
                extracted_aweme_id = match.group(1)
                
        if extracted_aweme_id:
            existing_douyin_video = DouyinVideo.query.filter_by(aweme_id=extracted_aweme_id).first()
            if existing_douyin_video:
                # 更新关联
                existing_douyin_video.video_file_id = file_id
                db.session.commit()
                logger.info(f"更新视频关联(直接下载): DouyinVideo (aweme_id: {extracted_aweme_id}) -> VideoFile (id: {file_id})")

        logger.info(f"视频下载成功（直接下载）: {download_path}, ID: {file_id}")
                
        # 启动后台处理
        try:
            # 启动缩略图生成
            thumbnail_thread = threading.Thread(
                target=generate_video_thumbnail,
                args=(str(upload_path), file_id)
            )
            thumbnail_thread.daemon = True
            thumbnail_thread.start()
            
            # 启动视频处理
            processing_thread = threading.Thread(
                target=auto_process_video,
                args=(file_id,)
            )
            processing_thread.daemon = True
            processing_thread.start()
        except Exception as e:
            logger.error(f"启动后处理任务失败: {str(e)}")
        
        return {
            "code": 200, 
            "message": "下载成功",
            "data": {
                "fileId": file_id,
                "filename": f"从{platform}下载的视频",
                "size": file_size,
                "mimeType": f"video/{file_ext}",
                "url": f"/api/videos/{file_id}",
                "thumbnail": f"/api/videos/{file_id}/thumbnail",
                "platform": platform,
                "originalUrl": video_url,
                "downloadPath": str(download_path)
            }
        }
    
    except Exception as e:
        logger.exception(f"直接下载处理失败: {str(e)}")
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



