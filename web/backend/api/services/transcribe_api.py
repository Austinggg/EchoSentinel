# transcribe_api.py
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from utils.HttpResponse import success_response, error_response
import os
import tempfile
import logging

transcribe_api = Blueprint('transcribe', __name__)

# 初始化转录器
from services.content_analysis.video_transcribe import VideoTranscriber  # 调整导入路径
transcriber = VideoTranscriber()

@transcribe_api.route('/api/transcribe/file', methods=['POST'])
def transcribe_file():
    """
    处理单个视频文件上传并返回转录结果
    ---
    tags:
      - 视频转录
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: 上传的视频文件
      - in: formData
        name: keep_audio
        type: boolean
        default: false
        description: 是否保留生成的音频文件
    responses:
      200:
        description: 转录结果
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            data:
              type: object
              properties:
                filename:
                  type: string
                  example: "example.mp4"
                chunks:
                  type: array
                  items:
                    type: object
                    properties:
                      start:
                        type: number
                      end:
                        type: number
                      text:
                        type: string
                full_text:
                  type: string
                audio_path:
                  type: string
                  nullable: true
            message:
              type: string
      400:
        description: 无效文件类型或参数错误
      500:
        description: 服务器处理错误
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
            
        response_data = {
            "filename": file.filename,
            "chunks": result.get("chunks", []),
            "full_text": result.get("full_text", ""),
            "audio_path": audio_path if keep_audio else None
        }
        
        return success_response(response_data)
    
    except Exception as e:
        logging.exception("文件处理异常")
        return error_response(500, f"处理失败: {str(e)}")

@transcribe_api.route('/api/transcribe/directory', methods=['POST'])
def transcribe_directory():
    """
    批量处理服务器本地目录的视频文件
    ---
    tags:
      - 视频转录
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - input_dir
          properties:
            input_dir:
              type: string
              example: "/var/videos/input"
              description: 服务器上的输入目录绝对路径
            output_dir:
              type: string
              example: "/var/videos/output"
              description: 服务器上的输出目录绝对路径（可选）
    responses:
      200:
        description: 处理任务已启动
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            data:
              type: object
              properties:
                processed_count:
                  type: integer
                output_dir:
                  type: string
            message:
              type: string
      400:
        description: 目录路径无效
      500:
        description: 服务器处理错误
    """
    try:
        data = request.json
        if not data or 'input_dir' not in data:
            return error_response(400, "缺少必要参数：input_dir")
        
        # 验证路径安全性
        input_dir = os.path.abspath(data['input_dir'])
        if not os.path.exists(input_dir):
            return error_response(400, "输入目录不存在")
        
        # 设置输出目录
        output_dir = data.get('output_dir')
        if not output_dir:
            output_dir = os.path.join(input_dir, "transcript_output")
        
        # 执行目录处理
        transcriber.process_directory(
            video_dir=input_dir,
            output_dir=output_dir
        )
        
        # 统计处理结果
        processed_files = [
            f for f in os.listdir(output_dir) 
            if f.endswith('.json')
        ]
        
        return success_response({
            "processed_count": len(processed_files),
            "output_dir": output_dir
        })
    
    except Exception as e:
        logging.exception("目录处理异常")
        return error_response(500, f"目录处理失败: {str(e)}")