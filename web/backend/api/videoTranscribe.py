from flask import Blueprint, request, jsonify
from api.videoUpload import get_video_file_path
from werkzeug.utils import secure_filename
from utils.database import db
from utils.HttpResponse import success_response, error_response
import os
import tempfile
import logging

from utils import HttpResponse

transcribe_api = Blueprint('transcribe', __name__)

# 初始化转录器
from services.content_analysis.video_transcribe import VideoTranscriber  # 调整导入路径
transcriber = VideoTranscriber()

@transcribe_api.route('/api/transcribe/file', methods=['POST'])
def transcribe_file():
    """
    处理单个视频文件上传并返回转录结果
    """
    try:
        # 检查文件上传
        if 'file' not in request.files:
            return error_response(400, "未上传文件")
        
        file = request.files['file']
        if file.filename == '':
            return error_response(400, "无效文件名")
        
        # 验证文件类型
        allowed_extensions = {'mp4', 'avi', 'mkv', 'mov', 'flv'}
        if '.' not in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return error_response(400, "不支持的文件类型")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        temp_video = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_video)
        
        # 处理参数
        keep_audio = request.form.get('keep_audio', 'false').lower() == 'true'
        
        # 执行转录
        result = transcriber.transcribe_video(temp_video)
        
        # 清理临时文件
        if result and not keep_audio:
            audio_path = os.path.splitext(temp_video)[0] + "_audio.wav"
            if os.path.exists(audio_path):
                os.remove(audio_path)
        os.remove(temp_video)
        
        if not result:
            return error_response(500, "视频转录失败")
            
        # 修复这一部分：正确处理全文
        response_data = {
            "filename": file.filename,
            "chunks": result.get("chunks", []),
            "full_text": result.get("text", ""),  # 修改这里，使用text字段
            "audio_path": audio_path if keep_audio else None
        }
        
        return success_response(response_data)
    
    except Exception as e:
        logging.exception("文件处理异常")
        return error_response(500, f"处理失败: {str(e)}")

"""通过ID转录视频"""
@transcribe_api.route("/api/videos/<file_id>/transcribe", methods=["POST"])
def transcribe_video_by_id(file_id):
    try:
        from utils.database import VideoFile, VideoTranscript
        
        # 查询数据库获取视频信息
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        if not video:
            return error_response(404, "视频不存在")
        
        # 获取视频文件路径
        video_path = get_video_file_path(file_id, video.extension)
        if not video_path or not video_path.exists():
            return error_response(404, "视频文件不存在")
        
        # 检查文件大小
        file_size = video_path.stat().st_size
        print(f"视频文件大小: {file_size}")
        if file_size == 0:
            return error_response(400, "视频文件为空")
        
        # 执行转录
        print(f"开始转录视频: {str(video_path)}")
        result = transcriber.transcribe_video(str(video_path))
        
        if not result:
            return error_response(500, "视频转录失败，请检查服务器日志")
            
        # 如果视频没有音频轨道
        if result.get("text", "") == "" and result.get("message") == "视频不包含音频轨道":
            response_data = {
                "video_id": file_id,
                "filename": video.filename,
                "chunks": [],
                "full_text": "视频不包含音频轨道",
                "duration": 0,
                "message": "视频不包含音频轨道，无法转录"
            }
            return success_response(response_data)  # 返回成功但内容为空
            
        
        # 构建响应数据
        response_data = {
            "video_id": file_id,
            "filename": video.filename,
            "chunks": result.get("chunks", []),
            "full_text": result.get("text", ""),
            "duration": result.get("duration", 0)
        }
        
        # 更新或创建转录记录
        save_transcript_to_db(file_id, result, video)
        
        return success_response(response_data)
        
    except Exception as e:
        print(f"视频转录失败: {str(e)}")
        return error_response(500, f"视频转录失败: {str(e)}")
    

# 添加一个函数用于保存转录结果到数据库
def save_transcript_to_db(video_id, result, video):
    """保存转录结果到数据库"""
    from utils.database import VideoTranscript, db
    
    if "text" in result:
        # 查找是否存在现有记录
        transcript = db.session.query(VideoTranscript).filter_by(video_id=video_id).first()
        
        if transcript:
            # 更新现有记录
            transcript.transcript = result["text"]
            transcript.chunks = result.get("chunks", [])
        else:
            # 创建新记录
            transcript = VideoTranscript(
                video_id=video_id,
                transcript=result["text"],
                chunks=result.get("chunks", [])
            )
            db.session.add(transcript)
        
        # 同时更新视频表中的字段以保持兼容性
        video.transcript = result["text"]
        db.session.commit()