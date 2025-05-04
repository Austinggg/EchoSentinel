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
            return_timestamps=True,
            chunk_length_s=20,  # 添加分块长度
            stride_length_s=4   # 添加帧长度
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
            
            # 修复时间戳问题
            if 'chunks' in result:
                last_end_time = 0
                valid_chunks = []
                
                for chunk in result['chunks']:
                    # 跳过空文本
                    if not chunk['text'].strip():
                        continue
                    
                    try:
                        # 获取原始时间戳值，防御性处理
                        orig_start, orig_end = chunk.get('timestamp', (None, None))
                        
                        # 创建新的时间戳值
                        new_start = last_end_time if orig_start is None or orig_start < last_end_time else orig_start
                        new_end = orig_end
                        
                        if new_end is None or new_end <= new_start:
                            # 估算结束时间(假设每秒5个单词)
                            word_count = len(chunk['text'].split())
                            duration = max(word_count / 5, 0.5)  # 至少0.5秒
                            new_end = new_start + duration
                        
                        # 替换为新的时间戳列表
                        chunk['timestamp'] = [new_start, new_end]
                        last_end_time = new_end
                        valid_chunks.append(chunk)
                    except Exception as chunk_error:
                        # 如果处理特定chunk时出错，记录错误并继续处理
                        print(f"处理chunk时出错: {chunk_error}，跳过此chunk")
                        continue
                
                # 替换原始chunks列表
                result['chunks'] = valid_chunks
            
            return jsonify(result)
        except Exception as e:
            print(f"转录处理出错: {e}")
            return jsonify({"error": str(e)}), 500