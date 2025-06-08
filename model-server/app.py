import os

import aigc_detection
from config import MODEL
from decision_service import init_decision, register_decision_routes
from digital_human_service import init_digital_human, register_digital_human_routes
from extensions import app, init_dataset
from train_service import init_training_service, register_training_routes
from whisper_service import init_whisper, register_whisper_routes

init_dataset(app)

# 创建模型目录
os.makedirs(MODEL["save_dir"], exist_ok=True)

# 导入和注册 whisper 服务
whisper_ready = init_whisper(app)
register_whisper_routes(app)

# 导入和注册 decision 服务
decision_ready = init_decision(app)
register_decision_routes(app)

# 导入和注册 training 服务
training_ready = init_training_service(app)
register_training_routes(app)

# 导入和注册 digital_human 服务
digital_human_ready = init_digital_human(app)
register_digital_human_routes(app)

# 健康检查路由
@app.route("/health", methods=["GET"])
def health_check():
    from flask import jsonify

    status = {
        "whisper": {"status": "healthy" if whisper_ready else "unhealthy"},
        "decision": {"status": "healthy" if decision_ready else "unhealthy"},
        "training": {"status": "healthy" if training_ready else "unhealthy"},
        "digital_human": {"status": "healthy" if digital_human_ready else "unhealthy"},
        "device": "cuda:0"
        if whisper_ready
        and hasattr(app, "whisper_device")
        and app.whisper_device == "cuda:0"
        else "cpu",
    }

    all_healthy = all([whisper_ready, decision_ready, training_ready, digital_human_ready])
    return jsonify(status), 200 if all_healthy else 503

app.register_blueprint(aigc_detection.bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, threaded=True, debug=True)