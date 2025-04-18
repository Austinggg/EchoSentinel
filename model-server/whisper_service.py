from flask import request, jsonify
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

def init_whisper(app):
    """初始化 Whisper 模型并设置应用上下文"""
    app.whisper_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_path = "mod/whisper-large-v3-turbo"
    
    try:
        # 加载模型和处理器
        app.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True  # 强制只使用本地文件
        ).to(app.whisper_device)
        
        app.whisper_processor = AutoProcessor.from_pretrained(model_path,local_files_only=True)
        
        app.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=app.whisper_model,
            tokenizer=app.whisper_processor.tokenizer,
            feature_extractor=app.whisper_processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=app.whisper_device,
            return_timestamps=True
        )
        
        app.whisper_ready = True
        print(f"Whisper模型成功加载自: {model_path}")
        return True
    except Exception as e:
        print(f"Whisper模型加载失败: {e}")
        app.whisper_ready = False
        return False

def register_whisper_routes(app):
    """注册 Whisper 相关的API路由"""
    
    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        if not getattr(app, "whisper_ready", False):
            return jsonify({"error": "Whisper模型服务未就绪"}), 503
        
        if 'audio' not in request.files:
            return jsonify({"error": "未上传音频文件"}), 400
        
        audio_file = request.files['audio']
        temp_audio = f"/tmp/{os.urandom(16).hex()}.wav"
        
        try:
            audio_file.save(temp_audio)
            result = app.whisper_pipe(temp_audio)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)