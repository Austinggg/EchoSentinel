from flask import Blueprint, request, jsonify
import requests
import logging
from utils.HttpResponse import success_response, error_response
from utils.database import ContentAnalysis, VideoFile, db

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 远程服务器地址
CLASSIFICATION_SERVICE_URL = "http://121.48.227.136:3000"

# 创建蓝图 - 更改名称为decision_api
decision_api = Blueprint('decision', __name__)

@decision_api.route('/api/decision/classify', methods=['POST'])
def classify_content():
    """
    调用远程服务器对内容进行分类
    """
    try:
        # 获取请求数据
        data = request.json
        if not data:
            return error_response(400, "请求数据为空")
        
        # 构建发送到远程服务器的请求
        remote_url = f"{CLASSIFICATION_SERVICE_URL}/classify"
        
        # 发送请求到远程服务器
        response = requests.post(remote_url, json=data, timeout=10)
        
        # 检查响应
        if response.status_code != 200:
            logger.error(f"远程分类服务返回错误: {response.status_code} - {response.text}")
            return error_response(response.status_code, f"远程服务器错误: {response.text}")
        
        # 处理响应结果
        result = response.json()
        
        # 构建返回数据
        response_data = {
            "classification": result,
            "input": data
        }
        
        return success_response(response_data)
        
    except requests.exceptions.Timeout:
        logger.error("连接远程分类服务超时")
        return error_response(503, "连接远程分类服务超时")
    except requests.exceptions.ConnectionError:
        logger.error("无法连接到远程分类服务")
        return error_response(503, "无法连接到远程分类服务")
    except Exception as e:
        logger.exception("内容分类请求处理异常")
        return error_response(500, f"处理失败: {str(e)}")

@decision_api.route('/api/decision/explain', methods=['GET'])
def get_explanation():
    """
    获取模型规则解释
    """
    try:
        # 获取参数
        threshold = request.args.get('threshold', '0.5')
        format_type = request.args.get('format', 'enhanced')
        
        # 构建远程请求URL
        remote_url = f"{CLASSIFICATION_SERVICE_URL}/explain?threshold={threshold}&format={format_type}"
        
        # 发送请求
        response = requests.get(remote_url, timeout=10)
        
        # 检查响应
        if response.status_code != 200:
            logger.error(f"获取规则解释失败: {response.status_code} - {response.text}")
            return error_response(response.status_code, f"远程服务器错误: {response.text}")
        
        # 返回解释结果
        result = response.json()
        return success_response(result)
        
    except Exception as e:
        logger.exception("获取规则解释异常")
        return error_response(500, f"处理失败: {str(e)}")

@decision_api.route('/api/decision/rules/edit', methods=['POST'])
def edit_rules():
    """
    编辑分类模型规则
    """
    try:
        # 获取请求数据
        data = request.json
        if not data:
            return error_response(400, "请求数据为空")
            
        # 构建远程请求
        remote_url = f"{CLASSIFICATION_SERVICE_URL}/rules/edit"
        
        # 发送编辑请求
        response = requests.post(remote_url, json=data, timeout=15)
        
        # 检查响应
        if response.status_code != 200:
            logger.error(f"编辑规则失败: {response.status_code} - {response.text}")
            return error_response(response.status_code, f"远程服务器错误: {response.text}")
            
        # 返回编辑结果
        result = response.json()
        return success_response(result)
        
    except Exception as e:
        logger.exception("编辑规则异常")
        return error_response(500, f"处理失败: {str(e)}")
@decision_api.route('/api/videos/<video_id>/classify-risk', methods=['POST'])
def classify_video_risk(video_id):
    """
    根据视频ID获取评估结果，进行风险分类并更新数据库
    
    Args:
        video_id: 视频ID
    
    Returns:
        分类结果和更新后的风险等级
    """
    try:
        # 查询数据库获取内容分析记录
        content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
        if not content_analysis:
            return error_response(404, f"未找到视频ID为 {video_id} 的内容分析")
        
        # 获取对应的视频文件记录，用于更新风险等级
        video_file = VideoFile.query.filter_by(id=video_id).first()
        if not video_file:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频文件记录")
        
        # 从评估结果构建特征向量，如果某项评估结果为空则使用默认值0.5
        features = [
            content_analysis.p1_score or 0.5,  # 背景信息充分性
            content_analysis.p2_score or 0.5,  # 背景信息准确性
            content_analysis.p3_score or 0.5,  # 内容完整性
            content_analysis.p4_score or 0.5,  # 意图正当性
            content_analysis.p5_score or 0.5,  # 发布者信誉
            content_analysis.p6_score or 0.5,  # 情感中立性
            content_analysis.p7_score or 0.5,  # 行为自主性
            content_analysis.p8_score or 0.5   # 信息一致性
        ]
        
        # 构建请求数据
        classify_data = {
            "features": features
        }
        
        # 调用分类API
        try:
            remote_url = f"{CLASSIFICATION_SERVICE_URL}/classify"
            response = requests.post(remote_url, json=classify_data, timeout=10)
            
            # 检查响应
            if response.status_code != 200:
                logger.error(f"分类服务返回错误: {response.status_code} - {response.text}")
                return error_response(response.status_code, f"远程服务器错误: {response.text}")
            
            # 处理分类结果
            result = response.json()
            
            # 从响应中提取数据 - 注意处理多层嵌套结构
            if 'data' in result and 'classification' in result['data']:
                classification = result['data']['classification'].get('class_name', 'unknown')
                probability = result['data']['classification'].get('probability', 0)
            else:
                # 尝试其他可能的结构
                classification = result.get('class_name', 'unknown')
                probability = result.get('probability', 0)
            
            # 记录数据结构，便于调试
            logger.info(f"分类API返回结构: {result}")
            logger.info(f"解析得到: classification={classification}, probability={probability}")
            
            # 保存旧的风险等级（用于返回响应）
            old_risk_level = video_file.risk_level
            
            # 更新VideoFile表的风险等级
            video_file.risk_level = classification
            
            # 同时更新ContentAnalysis表的风险等级和概率
            content_analysis.risk_level = classification
            content_analysis.risk_probability = probability
            
            # 提交所有更改
            db.session.commit()
            
            # 构建响应数据
            response_data = {
                "video_id": video_id,
                "classification": {
                    "risk_level": classification,
                    "probability": probability
                },
                "features": features,
                "previous_risk_level": old_risk_level,
                "updated": True
            }
            
            return success_response(response_data)
            
        except requests.exceptions.Timeout:
            logger.error("连接分类服务超时")
            return error_response(503, "连接分类服务超时")
        except requests.exceptions.ConnectionError:
            logger.error("无法连接到分类服务")
            return error_response(503, "无法连接到分类服务")
        
    except Exception as e:
        db.session.rollback()
        logger.exception(f"处理视频风险分类请求时发生错误: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")