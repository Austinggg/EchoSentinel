from flask import Flask, request, jsonify
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os

app = Flask(__name__)

# 初始化模型（保持常驻）
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_path = "mod/whisper-large-v3-turbo"  # 确认路径正确

# 加载模型和处理器
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )
    model_ready = True  # 标记模型是否加载成功
except Exception as e:
    print(f"模型加载失败: {e}")
    model_ready = False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    if model_ready:
        return jsonify({
            "status": "healthy",
            "device": device,
            "model_path": model_path
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "error": "模型未加载"
        }), 503

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not model_ready:
        return jsonify({"error": "服务未就绪"}), 503
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']
    temp_audio = f"/tmp/{os.urandom(16).hex()}.wav"
    
    try:
        audio_file.save(temp_audio)
        result = pipe(temp_audio)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, threaded=True)