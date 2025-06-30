import json
import logging
import requests
from pathlib import Path
from flask import Blueprint, request
from services.content_analysis.analysis_reporter import AnalysisReporter
from utils.database import ContentAnalysis, VideoFile, VideoTranscript, DigitalHumanDetection, db
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
    根据视频ID生成综合威胁分析报告
    整合内容分析、数字人检测、事实核查等多维度数据
    
    Args:
        video_id: 视频ID
    
    Returns:
        生成的分析报告
    """
    try:
        # 查询内容分析数据
        content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
        if not content_analysis:
            return error_response(404, f"未找到视频ID为 {video_id} 的内容分析")
        
        # 获取视频基本信息
        video_file = VideoFile.query.filter_by(id=video_id).first()
        if not video_file:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频文件记录")
        
        # 获取数字人检测数据（可选）
        digital_human_data = None
        try:
            digital_human = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
            if digital_human and digital_human.status == "completed":
                digital_human_data = {
                    "status": "completed",
                    "detection": digital_human.to_dict(),
                    "summary": {
                        "final_prediction": None,
                        "confidence": None,
                        "ai_probability": None,
                        "human_probability": None
                    }
                }
                
                # 确定最终结果优先级：comprehensive > overall > face
                if digital_human.comprehensive_ai_probability is not None:
                    digital_human_data["summary"].update({
                        "final_prediction": digital_human.comprehensive_prediction,
                        "confidence": digital_human.comprehensive_confidence,
                        "ai_probability": digital_human.comprehensive_ai_probability,
                        "human_probability": digital_human.comprehensive_human_probability
                    })
                elif digital_human.overall_ai_probability is not None:
                    digital_human_data["summary"].update({
                        "final_prediction": digital_human.overall_prediction,
                        "confidence": digital_human.overall_confidence,
                        "ai_probability": digital_human.overall_ai_probability,
                        "human_probability": digital_human.overall_human_probability
                    })
                elif digital_human.face_ai_probability is not None:
                    digital_human_data["summary"].update({
                        "final_prediction": digital_human.face_prediction,
                        "confidence": digital_human.face_confidence,
                        "ai_probability": digital_human.face_ai_probability,
                        "human_probability": digital_human.face_human_probability
                    })
                
                logger.info(f"获取到数字人检测数据 - 视频ID: {video_id}")
        except Exception as e:
            logger.warning(f"获取数字人检测数据失败: {str(e)}")
        
        # 获取事实核查数据（可选）
        fact_check_data = None
        try:
            transcript = VideoTranscript.query.filter_by(video_id=video_id).first()
            if transcript and transcript.fact_check_status == "completed":
                fact_check_data = {
                    "status": "completed",
                    "worth_checking": transcript.worth_checking,
                    "reason": transcript.worth_checking_reason,
                    "claims": transcript.claims,
                    "fact_check_results": transcript.fact_check_results,
                    "search_summary": transcript.search_summary
                }
                logger.info(f"获取到事实核查数据 - 视频ID: {video_id}")
        except Exception as e:
            logger.warning(f"获取事实核查数据失败: {str(e)}")
        
        # 获取规则解释
        try:
            rules_response = requests.get(f"{CLASSIFICATION_SERVICE_URL}/explain?format=raw", timeout=10)
            if rules_response.status_code != 200:
                logger.warning(f"获取规则解释失败: {rules_response.status_code}")
                dnf_rules = {"disjuncts": {}, "conjuncts": []}
            else:
                dnf_rules = rules_response.json().get("data", {"disjuncts": {}, "conjuncts": []})
        except Exception as e:
            logger.warning(f"获取规则解释时出错: {str(e)}")
            dnf_rules = {"disjuncts": {}, "conjuncts": []}
        
        # 准备内容分析数据
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
        
        # 生成综合威胁分析报告
        analysis_report = reporter.generate_comprehensive_report(
            video_id, data_instance, dnf_rules, digital_human_data, fact_check_data
        )
        
        if analysis_report:
            # 保存报告到数据库
            content_analysis.analysis_report = analysis_report
            db.session.commit()
            logger.info(f"成功生成并保存综合威胁分析报告 - 视频ID: {video_id}")
            
            return success_response({
                "video_id": video_id,
                "report": analysis_report,
                "updated": True,
                "data_sources": {
                    "content_analysis": True,
                    "digital_human_detection": digital_human_data is not None,
                    "fact_checking": fact_check_data is not None
                }
            })
        else:
            logger.error(f"综合威胁分析报告生成失败: {video_id}")
            return error_response(500, "威胁分析报告生成失败")
            
    except Exception as e:
        db.session.rollback()
        logger.exception(f"生成综合威胁分析报告时发生错误: {str(e)}")
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