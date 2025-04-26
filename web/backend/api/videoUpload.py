import os
from pathlib import Path
import datetime
import uuid
from werkzeug.utils import secure_filename

from flask import Blueprint, request, jsonify,send_from_directory,send_file
from sqlalchemy import select

from userAnalyse.function import cal_loss as userAnalyse_main
from utils.database import UserProfile, db
from utils.HttpResponse import HttpResponse

bp = Blueprint("video", __name__)

# 获取项目根目录的绝对路径
BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent.parent

# 使用绝对路径定义上传目录
UPLOAD_DIR = BASE_DIR / "uploads" / "videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 使用绝对路径定义缩略图目录
THUMBNAIL_DIR = BASE_DIR / "uploads" / "images"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mov', 'm4a', 'wav', 'webm', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""上传视频文件API，支持同时上传多个文件（最多3个），并且生成缩略图"""
@bp.route("/api/videos/upload", methods=["POST"])
def upload_video():
    # 检查是否有文件上传
    if 'file' not in request.files and 'files[]' not in request.files:
        return HttpResponse.error("没有接收到文件", 400)
    
    # 处理多文件上传
    if 'files[]' in request.files:
        files = request.files.getlist('files[]')
        # 限制最多上传3个文件
        if len(files) > 3:
            return HttpResponse.error("最多只能同时上传3个文件", 400)
    # 向后兼容单文件上传
    else:
        files = [request.files['file']]
    
    # 检查所有文件有效性
    for file in files:
        if file.filename == '':
            return HttpResponse.error("存在未选择的文件", 400)
        if not allowed_file(file.filename):
            return HttpResponse.error(f"不支持的文件类型，仅支持: {', '.join(ALLOWED_EXTENSIONS)}", 400)
    
    try:
        # 处理每个文件
        uploaded_files = []
        
        for file in files:
            # 生成唯一ID
            file_id = str(uuid.uuid4())
            
            # 保存原始文件名用于解析 (不经过secure_filename处理)
            original_raw_filename = file.filename
            
            # 获取安全文件名用于存储
            original_filename = secure_filename(file.filename)
            file_ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
            
            # 仅使用唯一ID作为文件名
            unique_filename = f"{file_id}.{file_ext}" if file_ext else file_id
            file_path = UPLOAD_DIR / unique_filename
            
            # 保存文件
            file.save(file_path)
            
            # 获取文件相关信息
            file_size = os.path.getsize(file_path)
            file_mime = file.content_type
            
            # 解析文件名中的信息 - 使用原始未处理的文件名
            publish_time, title, tags = parse_crawler_filename(original_raw_filename)
            
            # 打印调试信息
            print(f"解析结果: 时间={publish_time}, 标题={title}, 标签={tags}")
            
            # 保存到数据库
            from utils.database import VideoFile
            
            new_video = VideoFile(
                id=file_id,
                filename=title if title else original_filename,  # 使用解析后的标题
                extension=file_ext,
                size=file_size,
                mime_type=file_mime,
                user_id=None,
                upload_time=datetime.datetime.utcnow(),
                publish_time=publish_time,  # 添加发布时间
                tags=','.join(tags) if tags else None,  # 添加标签
                status="processing"
            )
            
            db.session.add(new_video)
            
            # 添加到上传文件列表
            uploaded_files.append({
                "fileId": file_id,
                "filename": new_video.filename,
                "size": file_size,
                "mimeType": file_mime,
                "url": f"/api/videos/{file_id}",
                "publishTime": publish_time.isoformat() if publish_time else None,
                "tags": tags
            })
        # 上传成功后，立即尝试生成缩略图（异步生成，不影响上传响应）
            try:
                # 仅对视频文件生成缩略图
                if file_mime.startswith('video/'):
                    # 生成缩略图（在后台执行，不阻塞响应）
                    import threading
                    thumbnail_thread = threading.Thread(
                        target=generate_video_thumbnail,
                        args=(str(file_path), file_id)
                    )
                    thumbnail_thread.daemon = True
                    thumbnail_thread.start()
            except Exception as thumb_error:
            # 缩略图生成失败不影响上传成功
                print(f"上传后自动生成缩略图失败: {str(thumb_error)}")
        # 提交所有文件到数据库
        db.session.commit()
        
        # 返回结果
        if len(uploaded_files) == 1:
            return HttpResponse.success(uploaded_files[0])
        else:
            return HttpResponse.success({
                "files": uploaded_files,
                "count": len(uploaded_files)
            })
        
    except Exception as e:
        db.session.rollback()
        return HttpResponse.error(f"上传失败: {str(e)}", 500)

"""通过id获取视频文件"""
@bp.route("/api/videos/<file_id>", methods=["GET"])
def get_video(file_id):
    try:
        from utils.database import VideoFile
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        
        if not video:
            return HttpResponse.error("文件不存在", 404)
        
        # 根据ID和扩展名构建文件路径
        video_path = get_video_file_path(file_id, video.extension)
        
        if not video_path or not video_path.exists():
            return HttpResponse.error("视频文件不存在", 404)
            
        print(f"返回视频文件: {video_path}")
        return send_file(
            video_path,
            mimetype=video.mime_type or 'video/mp4',
            as_attachment=False
        )
    
    except Exception as e:
        print(f"获取视频失败: {str(e)}")
        return HttpResponse.error(f"获取文件失败: {str(e)}", 500)
    
"""获取所有上传的视频列表"""
@bp.route("/api/videos/list", methods=["GET"])
def list_videos():
    try:
        from utils.database import VideoFile
        
        # 获取查询参数（只保留筛选参数，移除分页参数）
        status = request.args.get('status', None)
        search = request.args.get('search', None)
        
        # 构建查询
        query = db.session.query(VideoFile)
        
        # 应用过滤器
        if status:
            query = query.filter(VideoFile.status == status)
        
        if search:
            query = query.filter(VideoFile.filename.ilike(f'%{search}%'))
        
        # 按上传时间倒序排序
        query = query.order_by(VideoFile.upload_time.desc())
        
        # 获取所有符合条件的视频（不分页）
        videos = query.all()
        
        # 构建响应（简化结构，只保留total和items）
        result = {
            "total": len(videos),
            "items": []
        }
        
        for video in videos:
            # 解析标签字符串为列表
            tags_list = video.tags.split(',') if video.tags else []
            
            result["items"].append({
                "id": video.id,
                "title": video.filename,
                "cover": f"/api/videos/{video.id}/thumbnail",
                "summary": video.summary,
                "threatLevel": video.risk_level,
                "createTime": video.upload_time.strftime("%Y-%m-%d"),
                "publishTime": video.publish_time.strftime("%Y-%m-%d %H:%M:%S") if video.publish_time else None,
                "tags": tags_list,
                "status": video.status
            })
        
        return HttpResponse.success(result)
        
    except Exception as e:
        return HttpResponse.error(f"获取视频列表失败: {str(e)}", 500)

"""获取视频缩略图，如果不存在则生成"""
@bp.route("/api/videos/<file_id>/thumbnail", methods=["GET"])
def get_video_thumbnail(file_id):
    try:
        from utils.database import VideoFile
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        if not video:
            return HttpResponse.error("文件不存在", 404)
            
        # 缩略图路径
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{file_id}.jpg")
        
        # 如果缩略图已存在，直接返回
        if os.path.exists(thumbnail_path):
            print(f"返回已存在的缩略图: {thumbnail_path}")
            return send_file(thumbnail_path, mimetype='image/jpeg')
            
        # 如果缩略图不存在，尝试生成
        if generate_video_thumbnail(video.path, file_id):
            print(f"返回新生成的缩略图: {thumbnail_path}")
            return send_file(thumbnail_path, mimetype='image/jpeg')
        
        # 如果生成失败，返回默认图片
        default_thumbnail = os.path.join(os.path.dirname(__file__), "../static/default_thumbnail.jpg")
        print(f"返回默认缩略图: {default_thumbnail}")
        return send_file(default_thumbnail, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"获取缩略图失败: {str(e)}")
        return HttpResponse.error(f"获取缩略图失败: {str(e)}", 500)

"""通过URL存储视频文件"""
@bp.route("/api/videos/store-by-url", methods=["POST"])
def store_video_by_url():
    """通过URL存储视频文件"""
    try:
        data = request.json
        if not data or 'url' not in data:
            return HttpResponse.error("未提供视频URL", 400)
        
        video_url = data['url']
        # 这里可以添加代码从URL下载视频
        # 实现略复杂，可能需要使用requests或urllib库
        
        # TODO: 下载视频文件，获取相关信息并保存
        # 这部分可以根据实际需求实现
        
        return HttpResponse.success({
            "message": "功能尚未实现，请使用文件上传接口"
        })
        
    except Exception as e:
        return HttpResponse.error(f"通过URL存储视频失败: {str(e)}", 500)

# 添加生成缩略图的独立函数
def generate_video_thumbnail(video_path, file_id):
    """根据视频生成缩略图，并返回是否成功"""
    try:
        import cv2
        
        # 缩略图路径
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{file_id}.jpg")
        
        # 创建视频捕获对象
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
            
        # 获取视频的原始宽高
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 跳到视频1秒处（避免黑屏）
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            raise Exception("无法读取视频帧")
            
        # 根据原始宽高比例选择缩略图尺寸
        aspect_ratio = original_width / original_height if original_height != 0 else 1.0
        
        # 判断是横版还是竖版视频
        if aspect_ratio >= 1.0:  # 横版视频 (16:9)
            thumbnail_width = 480  # 保持宽度为480
            thumbnail_height = int(thumbnail_width / aspect_ratio)
        else:  # 竖版视频 (9:16)
            thumbnail_height = 480  # 保持高度为480
            thumbnail_width = int(thumbnail_height * aspect_ratio)
        
        # 调整尺寸
        frame = cv2.resize(frame, (thumbnail_width, thumbnail_height))
        
        # 保存缩略图
        cv2.imwrite(str(thumbnail_path), frame)
        
        # 释放资源
        cap.release()
        
        return True
    
    except Exception as e:
        print(f"生成缩略图失败: {str(e)}")
        return False
    
def get_video_file_path(video_id, extension=None):
    """根据视频ID和扩展名获取视频文件的绝对路径"""
    if extension:
        return UPLOAD_DIR / f"{video_id}.{extension}"
    
    # 如果没有提供扩展名，尝试遍历所有可能的扩展名
    for ext in ALLOWED_EXTENSIONS:
        path = UPLOAD_DIR / f"{video_id}.{ext}"
        if path.exists():
            return path
    
    # 如果找不到视频，返回None
    return None

def parse_crawler_filename(filename):
    """
    解析爬虫爬取的视频文件名
    格式: 2025-03-20 18-05-00_#俄罗斯海盐_#俄罗斯无碘盐_#无碘盐_#食用盐_video.mp4
    
    返回:
        publish_time: 发布时间 (datetime对象)
        title: 标题
        tags: 标签列表
    """
    try:
        # 打印原始文件名用于调试
        print(f"解析原始文件名: {filename}")
        
        # 移除文件扩展名
        base_filename = filename
        if '.' in base_filename:
            base_filename = base_filename.rsplit('.', 1)[0]
            
        # 移除末尾的_video (如果有)
        if base_filename.endswith('_video'):
            base_filename = base_filename[:-6]
        
        # 按下划线分割
        parts = base_filename.split('_')
        
        # 提取发布时间 (第一部分)
        datetime_str = parts[0]
        try:
            publish_time = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H-%M-%S')
        except ValueError:
            publish_time = None
            
        # 提取标签 (带#的部分)
        tags = []
        for part in parts:
            if part.startswith('#'):
                tags.append(part)
        
        # 提取标题 (除去时间和标签的所有部分)
        title_parts = []
        for i, part in enumerate(parts):
            if i == 0 and publish_time:  # 跳过时间部分
                continue
            if part.startswith('#'):  # 跳过标签部分
                continue
            title_parts.append(part)
        
        # 将标题部分用空格连接，如果为空则设为"无"
        title = ' '.join(title_parts)
        if not title.strip():  # 检查是否为空或只有空格
            title = "无"  # 设置默认标题
        
        print(f"解析结果 - 时间: {publish_time}, 标题: {title}, 标签: {tags}")
        return publish_time, title, tags
    
    except Exception as e:
        print(f"解析文件名失败: {str(e)}, 文件名: {filename}")
        return None, filename, []