from flask import send_from_directory

from api import auth, menu, user, userAnalyse, videoUpload
from api.extractAndSummary import extract_api  # 导入提取API蓝图
from api.logicAssessment import assessment_api  # 导入评估API蓝图
from api.videoTranscribe import transcribe_api  # 导入转录API蓝图
from api.decision import decision_api  # 导入决策API蓝图
from api.workflow import workflow_api  # 导入工作流API蓝图
from api.analysisReport import report_api  # 导入分析报告API蓝图
from api.douyin_tiktok_api import douyin_api  # 导入抖音API蓝图
from api.account import account_api # 导入测试API蓝图
from utils.database import init_dataset
from utils.extensions import app

init_dataset(app)


@app.route("/")
def index():
    return send_from_directory("dist", "index.html")



app.register_blueprint(transcribe_api)  # 注册转录API蓝图
app.register_blueprint(extract_api)  # 注册提取API蓝图
app.register_blueprint(assessment_api)  # 注册评估API蓝图
app.register_blueprint(workflow_api)  # 注册工作流API蓝图
app.register_blueprint(decision_api)  # 注册决策API蓝图
app.register_blueprint(report_api)  # 注册分析报告API蓝图
app.register_blueprint(douyin_api)  # 注册抖音API蓝图
app.register_blueprint(account_api)  # 注册测试API蓝图
app.register_blueprint(auth.bp)
app.register_blueprint(menu.bp)
# app.register_blueprint(test.bp)
app.register_blueprint(user.bp)
app.register_blueprint(userAnalyse.bp)
app.register_blueprint(videoUpload.video_api)
if __name__ == "__main__":
    app.run(debug=True, port=8000)  # 开启调试模式（包含热重载）
