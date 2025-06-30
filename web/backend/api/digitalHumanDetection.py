import datetime
import logging
import os
import threading
from pathlib import Path

import requests
from flask import Blueprint, request, current_app  # 添加 current_app
from utils.database import DigitalHumanDetection, VideoFile, db
from utils.HttpResponse import error_response, success_response
from utils.redis_client import get_redis_client, redis_error_handler
logger = logging.getLogger(__name__)
digital_human_api = Blueprint('digital_human', __name__)

# 数字人检测服务配置
DIGITAL_HUMAN_SERVICE_URL = "http://121.48.227.136:3000"
DETECTION_TIMEOUT = 1800  # 30分钟超时

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

# Redis相关工具函数
def get_task_key(video_id: str) -> str:
    return f"task:digital_human:{video_id}"

def get_task_channel(video_id: str) -> str:
    return f"task_updates:digital_human:{video_id}"

@redis_error_handler(fallback_return=False)
def update_task_status_redis(video_id: str, status_data: dict) -> bool:
    """更新Redis中的任务状态"""
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    task_key = get_task_key(video_id)
    return redis_client.set_task_status(task_key, status_data, expire_time=7200)  # 2小时过期

@redis_error_handler(fallback_return=False)
def update_task_progress_redis(video_id: str, progress: float, current_step: str = None) -> bool:
    """更新Redis中的任务进度"""
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    task_key = get_task_key(video_id)
    success = redis_client.update_task_progress(task_key, progress, current_step)
    
    # 发布进度更新消息
    if success:
        message = {
            "video_id": video_id,
            "progress": progress,
            "current_step": current_step,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        redis_client.publish_task_update(get_task_channel(video_id), message)
    
    return success



def async_detect_digital_human(app, video_id, detection_types, comprehensive):
    """异步执行数字人检测 - 支持独立检测步骤"""
    
    with app.app_context():
        task_key = get_task_key(video_id)
        
        try:
            # 初始化Redis任务状态
            initial_status = {
                "video_id": video_id,
                "status": "processing",
                "progress": 0,
                "current_step": "initializing",
                "detection_types": detection_types,
                "comprehensive": comprehensive,
                "started_at": datetime.datetime.utcnow().isoformat(),
                "estimated_duration": f"{len(detection_types) * 5}-{len(detection_types) * 8}分钟"
            }
            update_task_status_redis(video_id, initial_status)
            
            # 查询检测记录
            detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
            video = VideoFile.query.filter_by(id=video_id).first()
            
            if not detection or not video:
                raise Exception("检测记录或视频不存在")
            
            video_path = get_video_file_path(video_id, video.extension)
            new_results = {}  # 本次新检测的结果
            
            total_steps = len(detection_types) + (1 if comprehensive else 0)
            current_step = 0
            
            # 并行或串行执行检测（这里保持串行以简化错误处理）
            detection_methods = {
                'face': ('aigc/detect/face', 'face_detection'),
                'body': ('aigc/detect/body', 'body_detection'), 
                'overall': ('aigc/detect/overall', 'overall_detection')
            }
            
            completed_detections = 0  # 记录完成的检测数量
            
            for detection_type in detection_types:
                if detection_type not in detection_methods:
                    continue
                    
                endpoint, step_name = detection_methods[detection_type]
                current_step += 1
                
                logger.info(f"开始{detection_type}检测 - 视频ID: {video_id}")
                progress = int((current_step - 0.5) / total_steps * 90)  # 留10%给综合评估
                update_task_progress_redis(video_id, progress, step_name)
                
                # 更新对应模块状态为processing
                setattr(detection, f'{detection_type}_status', 'processing')
                setattr(detection, f'{detection_type}_started_at', datetime.datetime.utcnow())
                db.session.commit()
                
                try:
                    result = call_detection_service(endpoint, video_path)
                    
                    # 更新数据库对应字段
                    setattr(detection, f'{detection_type}_ai_probability', result['ai_probability'])
                    setattr(detection, f'{detection_type}_human_probability', result['human_probability'])
                    setattr(detection, f'{detection_type}_confidence', result['confidence'])
                    setattr(detection, f'{detection_type}_prediction', result['prediction'])
                    setattr(detection, f'{detection_type}_raw_results', result['raw_results'])
                    
                    # 更新模块状态为完成
                    setattr(detection, f'{detection_type}_status', 'completed')
                    setattr(detection, f'{detection_type}_completed_at', datetime.datetime.utcnow())
                    setattr(detection, f'{detection_type}_error_message', None)
                    
                    new_results[detection_type] = result
                    completed_detections += 1  # 增加完成计数
                    
                    # 更新进度
                    progress = int(current_step / total_steps * (90 if comprehensive else 100))
                    detection.progress = progress
                    detection.current_step = f"{detection_type}_completed"
                    db.session.commit()
                    update_task_progress_redis(video_id, progress, f"{detection_type}_completed")
                    
                    logger.info(f"{detection_type}检测完成 - 视频ID: {video_id}, 结果: {result['prediction']}")
                    
                except Exception as detection_error:
                    logger.error(f"{detection_type}检测失败 - 视频ID: {video_id}, 错误: {str(detection_error)}")
                    
                    # 更新模块状态为失败
                    setattr(detection, f'{detection_type}_status', 'failed')
                    setattr(detection, f'{detection_type}_error_message', str(detection_error))
                    db.session.commit()
                    
                    # 继续其他检测，不中断整个流程
                    continue
            
            # 检查是否所有请求的检测都已完成或失败
            requested_completed = 0
            for detection_type in detection_types:
                status = getattr(detection, f'{detection_type}_status', 'not_started')
                if status in ['completed', 'failed']:
                    requested_completed += 1
            
            # 综合评估逻辑 - 修复：只有在需要且有结果时才执行
            if comprehensive and completed_detections > 0:
                logger.info(f"开始综合评估 - 视频ID: {video_id}")
                update_task_progress_redis(video_id, 95, "comprehensive_analysis")
                
                # 更新综合评估状态为processing
                detection.comprehensive_status = 'processing'
                detection.comprehensive_started_at = datetime.datetime.utcnow()
                db.session.commit()
                
                try:
                    # 重新查询数据库获取所有可用的检测结果
                    detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
                    all_results = {}
                    
                    # 收集数据库中所有可用的检测结果
                    if detection.face_ai_probability is not None:
                        all_results['face'] = {
                            'ai_probability': detection.face_ai_probability,
                            'human_probability': detection.face_human_probability,
                            'confidence': detection.face_confidence,
                            'prediction': detection.face_prediction,
                            'raw_results': detection.face_raw_results
                        }
                    
                    if detection.body_ai_probability is not None:
                        all_results['body'] = {
                            'ai_probability': detection.body_ai_probability,
                            'human_probability': detection.body_human_probability,
                            'confidence': detection.body_confidence,
                            'prediction': detection.body_prediction,
                            'raw_results': detection.body_raw_results
                        }
                    
                    if detection.overall_ai_probability is not None:
                        all_results['overall'] = {
                            'ai_probability': detection.overall_ai_probability,
                            'human_probability': detection.overall_human_probability,
                            'confidence': detection.overall_confidence,
                            'prediction': detection.overall_prediction,
                            'raw_results': detection.overall_raw_results
                        }
                    
                    # 基于所有可用结果计算综合评估
                    if len(all_results) > 0:
                        weights = {"face": 0.3, "body": 0.2, "overall": 0.5}
                        available_weights = {dt: weights[dt] for dt in all_results.keys() if dt in weights}
                        
                        if available_weights:
                            total_weight = sum(available_weights.values())
                            
                            # 重新归一化权重
                            normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                            
                            weighted_ai_score = sum(all_results[dt]['ai_probability'] * normalized_weights[dt] 
                                                  for dt in normalized_weights.keys())
                            weighted_human_score = sum(all_results[dt]['human_probability'] * normalized_weights[dt] 
                                                     for dt in normalized_weights.keys())
                            
                            predictions = [all_results[dt]['prediction'] for dt in all_results.keys()]
                            
                            # 计算一致性
                            ai_votes = predictions.count("AI-Generated")
                            human_votes = predictions.count("Human")
                            consensus = len(set(predictions)) == 1  # 所有预测一致
                            
                            # 置信度计算
                            base_confidence = max(weighted_ai_score, weighted_human_score)
                            confidence_penalty = 0 if consensus else 0.1  # 不一致时降低置信度
                            final_confidence = max(0, min(1, base_confidence - confidence_penalty))
                            
                            # 更新综合评估结果
                            detection.comprehensive_ai_probability = weighted_ai_score
                            detection.comprehensive_human_probability = weighted_human_score
                            detection.comprehensive_confidence = final_confidence
                            detection.comprehensive_prediction = "AI-Generated" if weighted_ai_score > weighted_human_score else "Human"
                            detection.comprehensive_consensus = consensus
                            detection.comprehensive_votes = {"ai": ai_votes, "human": human_votes}
                            
                            # 更新综合评估状态为完成
                            detection.comprehensive_status = 'completed'
                            detection.comprehensive_completed_at = datetime.datetime.utcnow()
                            detection.comprehensive_error_message = None
                            
                            # 将综合评估结果也加入结果集
                            all_results['comprehensive'] = {
                                "ai_probability": weighted_ai_score,
                                "human_probability": weighted_human_score,
                                "confidence": final_confidence,
                                "prediction": detection.comprehensive_prediction,
                                "consensus": consensus,
                                "votes": {"ai": ai_votes, "human": human_votes},
                                "available_detections": list(all_results.keys())
                            }
                            
                            logger.info(f"综合评估完成 - 视频ID: {video_id}, 基于检测: {list(all_results.keys())}, 结果: {detection.comprehensive_prediction}")
                            
                            # 更新最终结果变量为所有结果
                            final_results = all_results
                        else:
                            # 如果没有有效的权重配置，使用新检测结果
                            final_results = new_results
                    else:
                        # 如果没有任何结果，使用新检测结果
                        final_results = new_results
                        
                except Exception as comprehensive_error:
                    logger.error(f"综合评估失败 - 视频ID: {video_id}, 错误: {str(comprehensive_error)}")
                    # 即使综合评估失败，也继续使用新检测结果
                    final_results = new_results
            else:
                final_results = new_results
            
            # 修复：正确判断任务完成条件
            if requested_completed == len(detection_types):
                if completed_detections > 0:
                    # 至少有一个检测成功完成，标记为完成
                    detection.status = "completed"
                    detection.completed_at = datetime.datetime.utcnow()
                    detection.progress = 100
                    detection.current_step = "completed"
                    detection.error_message = None
                    
                    # 更新Redis状态为完成
                    completed_status = {
                        "video_id": video_id,
                        "status": "completed",
                        "progress": 100,
                        "current_step": "completed",
                        "completed_at": datetime.datetime.utcnow().isoformat(),
                        "detection_types": detection_types,
                        "comprehensive": comprehensive,
                        "results": final_results if 'final_results' in locals() else new_results
                    }
                    update_task_status_redis(video_id, completed_status)
                    
                    db.session.commit()
                    logger.info(f"数字人检测全部完成 - 视频ID: {video_id}, 完成检测: {completed_detections}/{len(detection_types)}")
                else:
                    # 所有检测都失败
                    detection.status = "failed"
                    detection.error_message = "所有检测模块都失败"
                    detection.progress = 0
                    detection.current_step = "failed"
                    
                    # 更新Redis状态为失败
                    failed_status = {
                        "video_id": video_id,
                        "status": "failed",
                        "progress": 0,
                        "current_step": "failed",
                        "error_message": "所有检测模块都失败",
                        "failed_at": datetime.datetime.utcnow().isoformat()
                    }
                    update_task_status_redis(video_id, failed_status)
                    
                    db.session.commit()
                    logger.error(f"数字人检测全部失败 - 视频ID: {video_id}")
                    
        except Exception as e:
            # 更新数据库状态为失败
            try:
                detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
                if detection:
                    detection.status = "failed"
                    detection.error_message = str(e)
                    detection.progress = 0
                    detection.current_step = "failed"
                    db.session.commit()
            except Exception as db_error:
                logger.error(f"更新数据库失败状态时出错: {str(db_error)}")
            
            # 更新Redis状态为失败
            error_status = {
                "video_id": video_id,
                "status": "failed",
                "progress": 0,
                "current_step": "failed",
                "error_message": str(e),
                "failed_at": datetime.datetime.utcnow().isoformat()  # 修复拼写错误
            }
            update_task_status_redis(video_id, error_status)
            
            logger.error(f"异步数字人检测失败 - 视频ID: {video_id}, 错误: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/detect', methods=['POST'])
def detect_digital_human(video_id):
    """对指定视频进行数字人检测 - 支持选择性检测"""
    try:
        # 查询视频是否存在
        video = VideoFile.query.filter_by(id=video_id).first()
        if not video:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频")
        
        # 获取视频文件路径
        video_path = get_video_file_path(video_id, video.extension)
        if not video_path or not video_path.exists():
            return error_response(404, "视频文件不存在")
        
        # 首先检查Redis中的任务状态
        redis_client = get_redis_client()
        if redis_client:
            task_key = get_task_key(video_id)
            redis_status = redis_client.get_task_status(task_key)
            if redis_status and redis_status.get('status') == 'processing':
                return success_response({
                    "video_id": video_id,
                    "status": "processing",
                    "message": "检测任务正在进行中",
                    "progress": redis_status.get('progress', 0),
                    "current_step": redis_status.get('current_step', ''),
                    "estimated_duration": redis_status.get('estimated_duration', '15-20分钟'),
                    "source": "redis"
                })
        
        # 检查数据库中的检测记录
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if detection and detection.status == "processing":
            return success_response({
                "video_id": video_id,
                "status": "processing",
                "message": "检测任务正在进行中",
                "progress": getattr(detection, 'progress', 0),
                "current_step": getattr(detection, 'current_step', ''),
                "source": "database"
            })
        
        if not detection:
            detection = DigitalHumanDetection(video_id=video_id)
            db.session.add(detection)
        
        # 获取检测参数 - 允许选择性检测
        data = request.get_json() if request.content_type == 'application/json' else {}
        detection_types = data.get('types', ['face', 'body', 'overall'])
        comprehensive = data.get('comprehensive', True)
        
        # 验证检测类型
        valid_types = {'face', 'body', 'overall'}
        detection_types = [dt for dt in detection_types if dt in valid_types]
        
        if not detection_types:
            return error_response(400, "至少需要选择一种检测类型")
        
        # 如果只有一种检测类型，自动关闭综合评估
        if len(detection_types) == 1:
            comprehensive = False
        
        # 更新数据库状态为处理中
        detection.status = "processing"
        detection.error_message = None
        detection.progress = 0
        detection.current_step = "initializing"
        detection.started_at = datetime.datetime.utcnow()
        db.session.commit()
        
        # 在启动线程前获取应用实例
        app = current_app._get_current_object()
        
        # 启动异步任务（传递应用实例）
        thread = threading.Thread(
            target=async_detect_digital_human,
            args=(app, video_id, detection_types, comprehensive)  # 添加app参数
        )
        thread.daemon = True
        thread.start()
        
        return success_response({
            "video_id": video_id,
            "status": "processing",
            "message": f"检测任务已启动，将执行{len(detection_types)}个检测模块",
            "detection_types": detection_types,
            "comprehensive": comprehensive,
            "estimated_duration": f"{len(detection_types) * 5}-{len(detection_types) * 8}分钟",
            "task_key": get_task_key(video_id)
        })
        
    except Exception as e:
        db.session.rollback()
        logger.exception(f"启动数字人检测失败 - 视频ID: {video_id}")
        return error_response(500, f"启动检测失败: {str(e)}")
@digital_human_api.route('/api/videos/<video_id>/digital-human/status', methods=['GET'])
def get_detection_status(video_id):
    """获取检测进度状态 - Redis优先版本"""
    try:
        # 优先从Redis获取状态
        redis_client = get_redis_client()
        if redis_client:
            task_key = get_task_key(video_id)
            redis_status = redis_client.get_task_status(task_key)
            if redis_status:
                return success_response({
                    **redis_status,
                    "source": "redis"
                })
        
        # Redis中没有，从数据库获取
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if not detection:
            return error_response(404, "未找到检测记录")
        
        response_data = {
            "video_id": video_id,
            "status": detection.status,
            "progress": getattr(detection, 'progress', 0),
            "current_step": getattr(detection, 'current_step', ''),
            "started_at": detection.started_at.isoformat() if detection.started_at else None,
            "completed_at": detection.completed_at.isoformat() if detection.completed_at else None,
            "error_message": detection.error_message,
            "source": "database"
        }
        
        # 如果完成，返回结果
        if detection.status == "completed":
            response_data["results"] = detection.to_dict()
        
        return success_response(response_data)
        
    except Exception as e:
        logger.exception(f"获取检测状态失败 - 视频ID: {video_id}")
        return error_response(500, f"获取状态失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/cancel', methods=['POST'])
def cancel_detection(video_id):
    """取消检测任务"""
    try:
        # 更新数据库状态
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if detection and detection.status == "processing":
            detection.status = "cancelled"
            detection.error_message = "用户取消"
            detection.current_step = "cancelled"
            db.session.commit()
            
            # 更新Redis状态
            cancel_status = {
                "video_id": video_id,
                "status": "cancelled",
                "progress": 0,
                "current_step": "cancelled",
                "error_message": "用户取消",
                "cancelled_at": datetime.datetime.utcnow().isoformat()
            }
            update_task_status_redis(video_id, cancel_status)
            
            return success_response({"message": "检测任务已取消"})
        else:
            return error_response(400, "没有正在进行的检测任务")
    except Exception as e:
        return error_response(500, f"取消失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/reset', methods=['POST'])
def reset_detection_status(video_id):
    """重置检测状态"""
    try:
        # 清理数据库状态
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if detection:
            detection.status = "pending"
            detection.progress = 0
            detection.current_step = None
            detection.error_message = None
            detection.started_at = None
            detection.completed_at = None
            db.session.commit()
        
        # 清理Redis状态
        redis_client = get_redis_client()
        if redis_client:
            task_key = get_task_key(video_id)
            redis_client.delete_task_status(task_key)
        
        return success_response({"message": "检测状态已重置"})
        
    except Exception as e:
        return error_response(500, f"重置失败: {str(e)}")

@digital_human_api.route('/api/videos/<video_id>/digital-human/result', methods=['GET'])
def get_digital_human_result(video_id):
    """获取视频的数字人检测结果 - 支持部分完成状态"""
    try:
        # 查询视频是否存在
        video = VideoFile.query.filter_by(id=video_id).first()
        if not video:
            return error_response(404, f"未找到视频ID为 {video_id} 的视频")
        
        # 查询检测结果
        detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
        if not detection:
            return error_response(404, f"视频ID为 {video_id} 的数字人检测结果不存在")
        
        # 修改状态检查逻辑：支持部分完成
        if detection.status == "processing":
            # 检查是否有任何模块已完成
            has_completed_modules = any([
                detection.face_status == 'completed',
                detection.body_status == 'completed', 
                detection.overall_status == 'completed',
                detection.comprehensive_status == 'completed'
            ])
            
            if not has_completed_modules:
                return error_response(400, "检测任务仍在进行中，请等待完成后再获取结果")
                
        elif detection.status == "failed":
            # 检查是否有任何模块成功完成
            has_completed_modules = any([
                detection.face_status == 'completed',
                detection.body_status == 'completed', 
                detection.overall_status == 'completed',
                detection.comprehensive_status == 'completed'
            ])
            
            if not has_completed_modules:
                return error_response(400, f"检测任务失败: {detection.error_message}")
                
        elif detection.status not in ["completed", "processing"]:
            return error_response(400, f"检测任务未完成，当前状态: {detection.status}")
        
        # 构建响应数据
        response_data = {
            "video_id": video_id,
            "filename": video.filename,
            "status": detection.status,
            "detection": detection.to_dict(),
            "completed_at": detection.completed_at.isoformat() if detection.completed_at else None,
            "summary": {
                "final_prediction": None,
                "confidence": None,
                "ai_probability": None,
                "human_probability": None
            }
        }
        
        # 确定最终结果的优先级：comprehensive > overall > face
        if detection.comprehensive_ai_probability is not None:
            response_data["summary"]["final_prediction"] = detection.comprehensive_prediction
            response_data["summary"]["confidence"] = detection.comprehensive_confidence
            response_data["summary"]["ai_probability"] = detection.comprehensive_ai_probability
            response_data["summary"]["human_probability"] = detection.comprehensive_human_probability
        elif detection.overall_ai_probability is not None:
            response_data["summary"]["final_prediction"] = detection.overall_prediction
            response_data["summary"]["confidence"] = detection.overall_confidence
            response_data["summary"]["ai_probability"] = detection.overall_ai_probability
            response_data["summary"]["human_probability"] = detection.overall_human_probability
        elif detection.face_ai_probability is not None:
            response_data["summary"]["final_prediction"] = detection.face_prediction
            response_data["summary"]["confidence"] = detection.face_confidence
            response_data["summary"]["ai_probability"] = detection.face_ai_probability
            response_data["summary"]["human_probability"] = detection.face_human_probability
        
        return success_response(response_data)
        
    except Exception as e:
        logger.exception(f"获取数字人检测结果失败 - 视频ID: {video_id}")
        return error_response(500, f"获取检测结果失败: {str(e)}")

# 额外的辅助API - 批量查询状态
@digital_human_api.route('/api/digital-human/batch-status', methods=['POST'])
def get_batch_detection_status():
    """批量查询检测状态"""
    try:
        data = request.get_json()
        if not data or 'video_ids' not in data:
            return error_response(400, "请提供视频ID列表")
        
        video_ids = data['video_ids']
        if not isinstance(video_ids, list) or len(video_ids) == 0:
            return error_response(400, "视频ID列表不能为空")
        
        if len(video_ids) > 50:
            return error_response(400, "批量查询最多支持50个视频")
        
        results = []
        redis_client = get_redis_client()
        
        for video_id in video_ids:
            try:
                # 优先从Redis获取
                status_data = None
                if redis_client:
                    task_key = get_task_key(video_id)
                    redis_status = redis_client.get_task_status(task_key)
                    if redis_status:
                        status_data = {**redis_status, "source": "redis"}
                
                # Redis中没有，从数据库获取
                if not status_data:
                    detection = DigitalHumanDetection.query.filter_by(video_id=video_id).first()
                    if detection:
                        status_data = {
                            "video_id": video_id,
                            "status": detection.status,
                            "progress": getattr(detection, 'progress', 0),
                            "current_step": getattr(detection, 'current_step', ''),
                            "error_message": detection.error_message,
                            "source": "database"
                        }
                    else:
                        status_data = {
                            "video_id": video_id,
                            "status": "not_found",
                            "progress": 0,
                            "current_step": None,
                            "error_message": "未找到检测记录",
                            "source": "none"
                        }
                
                results.append(status_data)
                
            except Exception as e:
                results.append({
                    "video_id": video_id,
                    "status": "error",
                    "progress": 0,
                    "current_step": None,
                    "error_message": str(e),
                    "source": "error"
                })
        
        return success_response({
            "total": len(video_ids),
            "results": results
        })
        
    except Exception as e:
        logger.exception("批量查询检测状态失败")
        return error_response(500, f"批量查询失败: {str(e)}")

# 服务状态API
@digital_human_api.route('/api/digital-human/service-status', methods=['GET'])
def get_service_status():
    """获取数字人检测服务状态"""
    try:
        # 检查外部检测服务是否可用
        try:
            response = requests.get(f"{DIGITAL_HUMAN_SERVICE_URL}/status", timeout=5)
            external_service_available = response.status_code == 200
            external_service_info = response.json() if external_service_available else None
        except Exception as e:
            external_service_available = False
            external_service_info = {"error": str(e)}
        
        # 检查Redis状态
        redis_available = False
        redis_info = {}
        redis_client = get_redis_client()
        if redis_client:
            try:
                redis_client.redis_client.ping()
                redis_available = True
                redis_info = {"status": "connected"}
            except Exception as e:
                redis_info = {"error": str(e)}
        
        # 检查数据库状态
        database_available = False
        try:
            db.session.execute("SELECT 1")
            database_available = True
            database_info = {"status": "connected"}
        except Exception as e:
            database_info = {"error": str(e)}
        
        # 统计当前任务数量
        task_stats = {}
        try:
            processing_count = DigitalHumanDetection.query.filter_by(status="processing").count()
            completed_count = DigitalHumanDetection.query.filter_by(status="completed").count()
            failed_count = DigitalHumanDetection.query.filter_by(status="failed").count()
            
            task_stats = {
                "processing": processing_count,
                "completed": completed_count,
                "failed": failed_count,
                "total": processing_count + completed_count + failed_count
            }
        except Exception as e:
            task_stats = {"error": str(e)}
        
        return success_response({
            "service_available": external_service_available and database_available,
            "components": {
                "detection_service": {
                    "available": external_service_available,
                    "url": DIGITAL_HUMAN_SERVICE_URL,
                    "info": external_service_info
                },
                "redis": {
                    "available": redis_available,
                    "info": redis_info
                },
                "database": {
                    "available": database_available,
                    "info": database_info
                }
            },
            "task_statistics": task_stats
        })
        
    except Exception as e:
        return error_response(500, f"获取服务状态失败: {str(e)}")