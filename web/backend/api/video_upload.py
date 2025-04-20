import os
from pathlib import Path
import datetime
import uuid
from werkzeug.utils import secure_filename

from flask import Blueprint, request, jsonify,send_from_directory
from sqlalchemy import select

from userAnalyse.function import cal_loss as userAnalyse_main
from utils.database import UserProfile, db
from utils.HttpResponse import HttpResponse

bp = Blueprint("video", __name__)

# 确保上传目录存在
#修改对应的上传路径
UPLOAD_DIR = Path("/home/wl/EchoSentinel/uploads/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mov', 'm4a', 'wav', 'webm', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/api/videos/upload", methods=["POST"])
def upload_video():
    """上传视频文件API，支持同时上传多个文件（最多3个）"""
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
            
            # 获取原始文件的扩展名
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
            
            # 保存到数据库
            from utils.database import VideoFile
            
            # 获取当前用户ID（如果有认证系统）
            user_id = None  # 可以从会话或请求中获取
            
            new_video = VideoFile(
                id=file_id,
                filename=original_filename,
                path=str(file_path),
                size=file_size,
                mime_type=file_mime,
                user_id=user_id,
                upload_time=datetime.datetime.utcnow()
            )
            
            db.session.add(new_video)
            
            # 添加到上传文件列表
            uploaded_files.append({
                "fileId": file_id,
                "filename": original_filename,
                "size": file_size,
                "mimeType": file_mime,
                "url": f"/api/videos/{file_id}"
            })
        
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

@bp.route("/api/videos/<file_id>", methods=["GET"])
def get_video(file_id):
    """获取视频文件"""
    try:
        from utils.database import VideoFile
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        
        if not video:
            return HttpResponse.error("文件不存在", 404)
        
        directory = os.path.dirname(video.path)
        filename = os.path.basename(video.path)
        
        return send_from_directory(directory, filename, as_attachment=False)
    
    except Exception as e:
        return HttpResponse.error(f"获取文件失败: {str(e)}", 500)

@bp.route("/api/videos/<file_id>/analyze", methods=["POST"])
def analyze_video(file_id):
    """分析视频内容"""
    try:
        from utils.database import VideoFile
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        
        if not video:
            return HttpResponse.error("文件不存在", 404)
        
        # 调用userAnalyse模块分析视频
        analysis_result = userAnalyse_main(video.path)
        
        # 更新数据库中的分析结果
        video.analysis_result = analysis_result
        db.session.commit()
        
        return HttpResponse.success({
            "fileId": file_id,
            "analysis": analysis_result
        })
        
    except Exception as e:
        db.session.rollback()
        return HttpResponse.error(f"分析失败: {str(e)}", 500)

@bp.route("/api/videos/store-by-url", methods=["POST"])
def store_video_by_url():
    """通过URL存储视频"""
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