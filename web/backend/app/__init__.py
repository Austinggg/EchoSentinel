# app/__init__.py
from flask import Flask
from flask import send_from_directory
from .utils.database import init_dataset
from .utils.redis_client import init_redis

def create_app():
    # 从你现有的 utils.extensions 导入已配置的 app 实例
    from .utils.extensions import app
    
    # 初始化数据库和Redis（和你原来的代码一样）
    init_dataset(app)
    init_redis(app)
    @app.route("/")
    def index():
        return send_from_directory("dist", "index.html")
    
    # 注册所有蓝图
    register_blueprints(app)
    
    return app

def register_blueprints(app):
    """注册所有蓝图 - 直接从你的 app.py 复制过来"""    
    # 从你现有的 api 目录导入（保持原有导入路径）
    from .views import auth, user, userAnalyse, videoUpload
    from .views.account import account_api
    from .views.analysisReport import report_api
    from .views.decision import decision_api
    from .views.douyin_tiktok_api import douyin_api
    from .views.extractAndSummary import extract_api
    from .views.logicAssessment import assessment_api
    from .views.videoTranscribe import transcribe_api
    from .views.workflow import workflow_api
    from .views.analytics import analytics_api
    from .views.AISearch import search_api
    from .views.digitalHumanDetection import digital_human_api
    from .views.systemSettings import system_api
    
    # 注册蓝图（和你原来的代码完全一样，只是复制过来）
    app.register_blueprint(transcribe_api)
    app.register_blueprint(extract_api)
    app.register_blueprint(assessment_api)
    app.register_blueprint(workflow_api)
    app.register_blueprint(decision_api)
    app.register_blueprint(report_api)
    app.register_blueprint(douyin_api)
    app.register_blueprint(account_api)
    app.register_blueprint(analytics_api)
    app.register_blueprint(search_api)
    app.register_blueprint(auth.bp)
    app.register_blueprint(system_api)
    app.register_blueprint(user.bp)
    app.register_blueprint(userAnalyse.user_analyse_api)
    app.register_blueprint(videoUpload.video_api)
    app.register_blueprint(digital_human_api)