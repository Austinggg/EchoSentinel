from flask import Flask
import os

app = Flask(__name__)

# 导入和注册 whisper 服务
from whisper_service import init_whisper, register_whisper_routes
whisper_ready = init_whisper(app)
register_whisper_routes(app)

# 导入和注册 decision 服务
from decision_service import init_decision, register_decision_routes
decision_ready = init_decision(app)
register_decision_routes(app)

# 健康检查路由
@app.route('/health', methods=['GET'])
def health_check():
    from flask import jsonify
    
    status = {
        "whisper": {"status": "healthy" if whisper_ready else "unhealthy"},
        "decision": {"status": "healthy" if decision_ready else "unhealthy"},
        "device": "cuda:0" if whisper_ready and hasattr(app, "whisper_device") and app.whisper_device == "cuda:0" else "cpu"
    }
    
    if whisper_ready and decision_ready:
        return jsonify(status), 200
    else:
        return jsonify(status), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, threaded=True)