import json
import logging
import requests
from pathlib import Path
from flask import Blueprint, request
from services.content_analysis.analysis_reporter import AnalysisReporter
from utils.database import ContentAnalysis, VideoFile, db
from utils.HttpResponse import success_response, error_response

logger = logging.getLogger(__name__)
report_api = Blueprint('report', __name__)  

# 特征映射表，用于报告生成
FEATURE_MAP = {
    1: "背景信息充分性",
    2: "背景信息准确性",
    3: "内容完整性",
    4: "意图正当性", 
    5: "发布者信誉",
    6: "情感中立性",
    7: "行为自主性",
    8: "信息一致性"
}

# 分类服务URL
CLASSIFICATION_SERVICE_URL = "http://127.0.0.1:8000/api/decision"

# 实例化报告生成器的单例
reporter_instance = None

def get_reporter():
    """获取报告生成器单例"""
    global reporter_instance
    if reporter_instance is None:
        # 直接使用AnalysisReporter的配置加载机制
        reporter_instance = AnalysisReporter(feature_map=FEATURE_MAP)
        logger.info("初始化报告生成器")
    return reporter_instance

@report_api.route('/api/videos/<video_id>/generate-report', methods=['POST'])
def generate_video_analysis_report(video_id):
    """
    根据视频ID生成分析报告并保存到数据库
    
    Args:
        video_id: 视频ID
    
    Returns:
        生成的分析报告
    """
    try:
        # 查询数据库获取内容分析记录
        content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
        if not content_analysis:
            return error_response(404, f"未找到视频ID为 {video_id} 的内容分析")
        
        # 获取视频基本信息
        video_file = VideoFile.query.filter_by(id=video_id).first()
        if not video_file:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频文件记录")
        
        # 获取规则解释
        try:
            rules_response = requests.get(f"{CLASSIFICATION_SERVICE_URL}/explain?format=raw", timeout=10)
            if rules_response.status_code != 200:
                logger.warning(f"获取规则解释失败: {rules_response.status_code}")
                return error_response(rules_response.status_code, "获取规则解释失败")
            
            dnf_rules = rules_response.json().get("data", {})
            if not dnf_rules:
                logger.warning("获取的规则解释为空")
                dnf_rules = {"disjuncts": {}, "conjuncts": []}
                
        except Exception as e:
            logger.exception("获取规则解释时出错")
            dnf_rules = {"disjuncts": {}, "conjuncts": []}
        
        # 准备数据实例
        data_instance = {
            "message": content_analysis.summary or "",
            "general": {
                "P1": content_analysis.p1_score or 0.5,
                "P2": content_analysis.p2_score or 0.5,
                "P3": content_analysis.p3_score or 0.5,
                "P4": content_analysis.p4_score or 0.5,
                "P5": content_analysis.p5_score or 0.5,
                "P6": content_analysis.p6_score or 0.5,
                "P7": content_analysis.p7_score or 0.5,
                "P8": content_analysis.p8_score or 0.5
            }
        }
        
        # 获取报告生成器实例
        reporter = get_reporter()
        
        # 生成报告
        analysis_report = reporter.generate_report(data_instance, dnf_rules)
        
        if analysis_report:
            # 保存报告到数据库
            content_analysis.analysis_report = analysis_report
            db.session.commit()
            logger.info(f"成功生成并保存视频 {video_id} 的分析报告")
            
            return success_response({
                "video_id": video_id,
                "report": analysis_report,
                "updated": True
            })
        else:
            logger.error(f"报告生成服务未返回有效内容: {video_id}")
            return error_response(500, "报告生成失败")
            
    except Exception as e:
        db.session.rollback()
        logger.exception(f"生成分析报告时发生错误: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")

@report_api.route('/api/videos/<video_id>/report', methods=['GET'])
def get_video_analysis_report(video_id):
    """
    获取指定视频的分析报告
    
    Args:
        video_id: 视频ID
    
    Returns:
        视频分析报告内容
    """
    try:
        # 查询数据库获取内容分析记录
        content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
        if not content_analysis:
            return error_response(404, f"未找到视频ID为 {video_id} 的内容分析")
            
        # 检查是否已有报告
        if not content_analysis.analysis_report:
            # 增加自动生成报告功能
            if request.args.get('auto_generate') == 'true':
                logger.info(f"报告不存在，自动触发生成，视频ID: {video_id}")
                # 调用生成报告API
                generate_result = generate_video_analysis_report(video_id)
                if isinstance(generate_result, tuple) and generate_result[1] != 200:
                    return generate_result
                # 重新获取更新后的内容分析记录
                content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
            else:
                return error_response(404, f"视频ID为 {video_id} 的分析报告尚未生成")
            
        # 构建响应数据，包含更多相关信息
        video_data = {
            "video_id": video_id,
            "report": content_analysis.analysis_report,
            "risk_level": content_analysis.risk_level,
            "risk_probability": content_analysis.risk_probability,
            "scores": {
                "background_sufficiency": content_analysis.p1_score,
                "background_accuracy": content_analysis.p2_score,
                "content_completeness": content_analysis.p3_score,
                "intention_legitimacy": content_analysis.p4_score,
                "publisher_credibility": content_analysis.p5_score,
                "emotional_neutrality": content_analysis.p6_score,
                "behavior_autonomy": content_analysis.p7_score,
                "information_consistency": content_analysis.p8_score
            },
            "created_at": content_analysis.updated_at.isoformat() if content_analysis.updated_at else None
        }
        
        return success_response(video_data)
        
    except Exception as e:
        logger.exception(f"获取分析报告时发生错误: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")