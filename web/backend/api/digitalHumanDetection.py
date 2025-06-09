import datetime
import logging
import os
from pathlib import Path

import requests
from flask import Blueprint, request

from utils.database import DigitalHumanDetection, VideoFile, db
from utils.HttpResponse import error_response, success_response

logger = logging.getLogger(__name__)
digital_human_api = Blueprint('digital_human', __name__)

# 数字人检测服务配置
DIGITAL_HUMAN_SERVICE_URL = "http://121.48.227.136:3000"
DETECTION_TIMEOUT = 300  # 5分钟超时

# 获取视频文件路径函数 (复用videoUpload.py中的逻辑)
def get_video_file_path(video_id, extension=None):
    """根据视频ID和扩展名获取视频文件的绝对路径"""
    BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads" / "videos"
    ALLOWED_EXTENSIONS = {"mp3", "mp4", "mov", "m4a", "wav", "webm", "avi", "mkv"}
    
    if extension:
        return UPLOAD_DIR / f"{video_id}.{extension}"

    # 如果没有提供扩展名，尝试遍历所有可能的扩展名
    for ext in ALLOWED_EXTENSIONS:
        path = UPLOAD_DIR / f"{video_id}.{ext}"
        if path.exists():
            return path

    return None

def call_detection_service(endpoint, video_path):
    """调用数字人检测服务"""
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            response = requests.post(
                f"{DIGITAL_HUMAN_SERVICE_URL}/{endpoint}",
                files=files,
                timeout=DETECTION_TIMEOUT
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('result')
            else:
                raise Exception(result.get('error', '检测服务返回错误'))
        else:
            raise Exception(f"检测服务HTTP错误: {response.status_code}")
            
    except requests.exceptions.Timeout:
        raise Exception("检测服务请求超时")
    except requests.exceptions.ConnectionError:
        raise Exception("无法连接到检测服务")
    except Exception as e:
        raise Exception(f"调用检测服务失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/detect', methods=['POST'])
def detect_digital_human(video_id):
    """对指定视频进行数字人检测"""
    try:
        # 查询视频是否存在
        video = VideoFile.query.filter_by(id=video_id).first()
        if not video:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频")
        
        # 获取视频文件路径
        video_path = get_video_file_path(video_id, video.extension)
        if not video_path or not video_path.exists():
            return error_response(404, "视频文件不存在")
        
        # 检查是否已存在检测记录
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if not detection:
            detection = DigitalHumanDetection(video_id=video_id)
            db.session.add(detection)
        
        # 更新状态为处理中
        detection.status = "processing"
        detection.error_message = None
        db.session.commit()
        
        # 获取检测类型参数
        detection_types = request.json.get('types', ['face', 'body', 'overall']) if request.json else ['face', 'body', 'overall']
        comprehensive = request.json.get('comprehensive', True) if request.json else True
        
        results = {}
        
        try:
            # 执行各项检测
            if 'face' in detection_types:
                logger.info(f"开始面部检测 - 视频ID: {video_id}")
                face_result = call_detection_service('aigc/detect/face', video_path)
                
                detection.face_ai_probability = face_result['ai_probability']
                detection.face_human_probability = face_result['human_probability']
                detection.face_confidence = face_result['confidence']
                detection.face_prediction = face_result['prediction']
                detection.face_raw_results = face_result['raw_results']
                
                results['face'] = face_result
                logger.info(f"面部检测完成 - 视频ID: {video_id}, 结果: {face_result['prediction']}")
            
            if 'body' in detection_types:
                logger.info(f"开始躯体检测 - 视频ID: {video_id}")
                body_result = call_detection_service('aigc/detect/body', video_path)
                
                detection.body_ai_probability = body_result['ai_probability']
                detection.body_human_probability = body_result['human_probability']
                detection.body_confidence = body_result['confidence']
                detection.body_prediction = body_result['prediction']
                detection.body_raw_results = body_result['raw_results']
                
                results['body'] = body_result
                logger.info(f"躯体检测完成 - 视频ID: {video_id}, 结果: {body_result['prediction']}")
            
            if 'overall' in detection_types:
                logger.info(f"开始整体检测 - 视频ID: {video_id}")
                overall_result = call_detection_service('aigc/detect/overall', video_path)
                
                detection.overall_ai_probability = overall_result['ai_probability']
                detection.overall_human_probability = overall_result['human_probability']
                detection.overall_confidence = overall_result['confidence']
                detection.overall_prediction = overall_result['prediction']
                detection.overall_raw_results = overall_result['raw_results']
                
                results['overall'] = overall_result
                logger.info(f"整体检测完成 - 视频ID: {video_id}, 结果: {overall_result['prediction']}")
            
            # 如果需要综合评估且所有检测都完成
            if comprehensive and len(detection_types) >= 2:
                logger.info(f"开始综合评估 - 视频ID: {video_id}")
                
                # 计算综合结果
                weights = {"face": 0.3, "body": 0.2, "overall": 0.5}
                total_weight = sum(weights[dt] for dt in detection_types if dt in weights)
                
                weighted_ai_score = 0
                weighted_human_score = 0
                predictions = []
                
                for dt in detection_types:
                    if dt in results and dt in weights:
                        weight = weights[dt] / total_weight  # 归一化权重
                        weighted_ai_score += results[dt]['ai_probability'] * weight
                        weighted_human_score += results[dt]['human_probability'] * weight
                        predictions.append(results[dt]['prediction'])
                
                # 投票统计
                ai_votes = predictions.count("AI-Generated")
                human_votes = predictions.count("Human")
                consensus = ai_votes >= len(predictions) / 2
                
                # 一致性惩罚
                confidence_penalty = 0.1 if abs(ai_votes - human_votes) == 1 else 0
                final_confidence = max(weighted_ai_score, weighted_human_score) - confidence_penalty
                
                detection.comprehensive_ai_probability = weighted_ai_score
                detection.comprehensive_human_probability = weighted_human_score
                detection.comprehensive_confidence = max(0, min(1, final_confidence))
                detection.comprehensive_prediction = "AI-Generated" if consensus else "Human"
                detection.comprehensive_consensus = consensus
                detection.comprehensive_votes = {"ai": ai_votes, "human": human_votes}
                
                results['comprehensive'] = {
                    "ai_probability": weighted_ai_score,
                    "human_probability": weighted_human_score,
                    "confidence": detection.comprehensive_confidence,
                    "prediction": detection.comprehensive_prediction,
                    "consensus": consensus,
                    "votes": {"ai": ai_votes, "human": human_votes}
                }
                
                logger.info(f"综合评估完成 - 视频ID: {video_id}, 结果: {detection.comprehensive_prediction}")
            
            # 更新VideoFile表中的数字人概率字段
            if 'comprehensive' in results:
                video.digital_human_probability = results['comprehensive']['ai_probability']
            elif 'overall' in results:
                video.digital_human_probability = results['overall']['ai_probability']
            elif 'face' in results:
                video.digital_human_probability = results['face']['ai_probability']
            
            # 标记检测完成
            detection.status = "completed"
            detection.completed_at = datetime.datetime.utcnow()
            db.session.commit()
            
            logger.info(f"数字人检测全部完成 - 视频ID: {video_id}")
            
            return success_response({
                "video_id": video_id,
                "status": "completed",
                "results": results,
                "detection_types": detection_types,
                "comprehensive": comprehensive
            })
            
        except Exception as detection_error:
            # 标记检测失败
            detection.status = "failed"
            detection.error_message = str(detection_error)
            db.session.commit()
            
            logger.error(f"数字人检测失败 - 视频ID: {video_id}, 错误: {str(detection_error)}")
            raise detection_error
            
    except Exception as e:
        db.session.rollback()
        logger.exception(f"数字人检测请求处理失败 - 视频ID: {video_id}")
        return error_response(500, f"数字人检测失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/result', methods=['GET'])
def get_digital_human_result(video_id):
    """获取视频的数字人检测结果"""
    try:
        # 查询视频是否存在
        video = VideoFile.query.filter_by(id=video_id).first()
        if not video:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频")
        
        # 查询检测结果
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if not detection:
            return error_response(404, f"视频ID为 {video_id} 的数字人检测结果不存在")
        
        # 构建响应数据
        response_data = {
            "video_id": video_id,
            "filename": video.filename,
            "detection": detection.to_dict()
        }
        
        return success_response(response_data)
        
    except Exception as e:
        logger.exception(f"获取数字人检测结果失败 - 视频ID: {video_id}")
        return error_response(500, f"获取检测结果失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/face', methods=['POST'])
def detect_face_only(video_id):
    """仅进行面部检测"""
    try:
        video = VideoFile.query.filter_by(id=video_id).first()
        if not video:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频")
        
        video_path = get_video_file_path(video_id, video.extension)
        if not video_path or not video_path.exists():
            return error_response(404, "视频文件不存在")
        
        # 执行面部检测
        face_result = call_detection_service('aigc/detect/face', video_path)
        
        # 更新或创建检测记录
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if not detection:
            detection = DigitalHumanDetection(video_id=video_id)
            db.session.add(detection)
        
        detection.face_ai_probability = face_result['ai_probability']
        detection.face_human_probability = face_result['human_probability']
        detection.face_confidence = face_result['confidence']
        detection.face_prediction = face_result['prediction']
        detection.face_raw_results = face_result['raw_results']
        detection.status = "completed"
        detection.completed_at = datetime.datetime.utcnow()
        
        db.session.commit()
        
        return success_response({
            "video_id": video_id,
            "type": "face",
            "result": face_result
        })
        
    except Exception as e:
        logger.exception(f"面部检测失败 - 视频ID: {video_id}")
        return error_response(500, f"面部检测失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/body', methods=['POST'])
def detect_body_only(video_id):
    """仅进行躯体检测"""
    try:
        video = VideoFile.query.filter_by(id=video_id).first()
        if not video:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频")
        
        video_path = get_video_file_path(video_id, video.extension)
        if not video_path or not video_path.exists():
            return error_response(404, "视频文件不存在")
        
        # 执行躯体检测
        body_result = call_detection_service('aigc/detect/body', video_path)
        
        # 更新或创建检测记录
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if not detection:
            detection = DigitalHumanDetection(video_id=video_id)
            db.session.add(detection)
        
        detection.body_ai_probability = body_result['ai_probability']
        detection.body_human_probability = body_result['human_probability']
        detection.body_confidence = body_result['confidence']
        detection.body_prediction = body_result['prediction']
        detection.body_raw_results = body_result['raw_results']
        detection.status = "completed"
        detection.completed_at = datetime.datetime.utcnow()
        
        db.session.commit()
        
        return success_response({
            "video_id": video_id,
            "type": "body",
            "result": body_result
        })
        
    except Exception as e:
        logger.exception(f"躯体检测失败 - 视频ID: {video_id}")
        return error_response(500, f"躯体检测失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/overall', methods=['POST'])
def detect_overall_only(video_id):
    """仅进行整体检测"""
    try:
        video = VideoFile.query.filter_by(id=video_id).first()
        if not video:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频")
        
        video_path = get_video_file_path(video_id, video.extension)
        if not video_path or not video_path.exists():
            return error_response(404, "视频文件不存在")
        
        # 执行整体检测
        overall_result = call_detection_service('aigc/detect/overall', video_path)
        
        # 更新或创建检测记录
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if not detection:
            detection = DigitalHumanDetection(video_id=video_id)
            db.session.add(detection)
        
        detection.overall_ai_probability = overall_result['ai_probability']
        detection.overall_human_probability = overall_result['human_probability']
        detection.overall_confidence = overall_result['confidence']
        detection.overall_prediction = overall_result['prediction']
        detection.overall_raw_results = overall_result['raw_results']
        detection.status = "completed"
        detection.completed_at = datetime.datetime.utcnow()
        
        db.session.commit()
        
        return success_response({
            "video_id": video_id,
            "type": "overall",
            "result": overall_result
        })
        
    except Exception as e:
        logger.exception(f"整体检测失败 - 视频ID: {video_id}")
        return error_response(500, f"整体检测失败: {str(e)}")

@digital_human_api.route('/api/digital-human/status', methods=['GET'])
def get_service_status():
    """获取数字人检测服务状态"""
    try:
        # 检查服务是否可用
        response = requests.get(f"{DIGITAL_HUMAN_SERVICE_URL}/aigc/status", timeout=10)
        
        if response.status_code == 200:
            service_data = response.json()
            return success_response({
                "service_available": True,
                "service_url": DIGITAL_HUMAN_SERVICE_URL,
                "service_info": service_data
            })
        else:
            return success_response({
                "service_available": False,
                "service_url": DIGITAL_HUMAN_SERVICE_URL,
                "error": f"服务返回状态码: {response.status_code}"
            })
            
    except Exception as e:
        return success_response({
            "service_available": False,
            "service_url": DIGITAL_HUMAN_SERVICE_URL,
            "error": str(e)
        })

@digital_human_api.route('/api/digital-human/batch', methods=['POST'])
def batch_detect():
    """批量数字人检测"""
    try:
        data = request.get_json()
        if not data or 'video_ids' not in data:
            return error_response(400, "请提供视频ID列表")
        
        video_ids = data['video_ids']
        detection_types = data.get('types', ['face', 'body', 'overall'])
        comprehensive = data.get('comprehensive', True)
        
        if not isinstance(video_ids, list) or len(video_ids) == 0:
            return error_response(400, "视频ID列表不能为空")
        
        if len(video_ids) > 10:
            return error_response(400, "批量检测最多支持10个视频")
        
        results = []
        failed_videos = []
        
        for video_id in video_ids:
            try:
                # 模拟调用单个检测API
                # 这里可以直接调用detect_digital_human函数或通过内部API调用
                # 为简化，我们记录需要处理的视频
                video = VideoFile.query.filter_by(id=video_id).first()
                if video:
                    results.append({
                        "video_id": video_id,
                        "status": "queued",
                        "message": "已加入检测队列"
                    })
                else:
                    failed_videos.append({
                        "video_id": video_id,
                        "error": "视频不存在"
                    })
                    
            except Exception as e:
                failed_videos.append({
                    "video_id": video_id,
                    "error": str(e)
                })
        
        return success_response({
            "total": len(video_ids),
            "queued": len(results),
            "failed": len(failed_videos),
            "results": results,
            "failed_videos": failed_videos,
            "detection_types": detection_types,
            "comprehensive": comprehensive,
            "message": "批量检测任务已提交，请通过结果查询API获取进度"
        })
        
    except Exception as e:
        logger.exception("批量数字人检测失败")
        return error_response(500, f"批量检测失败: {str(e)}")