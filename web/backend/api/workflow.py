import time
import threading
import logging
import datetime
import requests
from flask import Blueprint, request, jsonify, current_app
from utils.database import db, VideoFile, VideoProcessingTask
from utils.HttpResponse import success_response, error_response

workflow_api = Blueprint('workflow', __name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('workflow')

# 定义处理步骤
PROCESSING_STEPS = [
    {
        "id": "transcription",
        "name": "视频转录",
        "endpoint": "/api/videos/{video_id}/transcribe",
        "method": "POST"
    },
    {
        "id": "extract",
        "name": "信息提取",
        "endpoint": "/api/extract/video/{video_id}",
        "method": "POST",
        "depends_on": "transcription"
    },
    {
        "id": "summary",
        "name": "生成摘要",
        "endpoint": "/api/summary/video/{video_id}",
        "method": "POST",
        "depends_on": "extract"
    },
    {
        "id": "assessment",
        "name": "内容评估",
        "endpoint": "/api/videos/{video_id}/assess",
        "method": "POST",
        "depends_on": "extract"
    },
    {
        "id": "classify",
        "name": "风险分类",
        "endpoint": "/api/videos/{video_id}/classify-risk",
        "method": "POST",
        "depends_on": "assessment"
    },
    {
        "id": "report",
        "name": "威胁报告",
        "endpoint": "/api/videos/{video_id}/generate-report",
        "method": "POST",
        "depends_on": "classify"
    }
]

@workflow_api.route('/api/videos/<video_id>/process', methods=['POST'])
def start_video_processing(video_id):
    """启动视频处理流程"""
    try:
        # 检查视频是否存在
        video = db.session.query(VideoFile).filter(VideoFile.id == video_id).first()
        if not video:
            return error_response(404, "视频不存在")
        
        # 获取要执行的步骤
        steps_to_run = request.json.get('steps', [step['id'] for step in PROCESSING_STEPS])
        force = request.json.get('force', False)  # 是否强制重新处理
        
        # 创建或更新任务记录
        for step_id in steps_to_run:
            # 查找步骤定义
            step_def = next((s for s in PROCESSING_STEPS if s['id'] == step_id), None)
            if not step_def:
                continue  # 跳过未定义的步骤
                
            # 检查现有任务
            task = db.session.query(VideoProcessingTask).filter(
                VideoProcessingTask.video_id == video_id,
                VideoProcessingTask.task_type == step_id
            ).first()
            
            if not task:
                # 创建新任务
                task = VideoProcessingTask(
                    video_id=video_id,
                    task_type=step_id,
                    status="pending",
                    progress=0.0
                )
                db.session.add(task)
            elif force or task.status == 'failed':
                # 重置现有任务
                task.status = "pending"
                task.progress = 0.0
                task.started_at = None
                task.completed_at = None
                task.error = None
        
        db.session.commit()
        
        # 启动后台处理线程
        thread = threading.Thread(
            target=process_video_workflow,
            args=(video_id, steps_to_run, current_app._get_current_object())
        )
        thread.daemon = True
        thread.start()
        
        return success_response({
            "message": "视频处理流程已启动",
            "video_id": video_id,
            "tasks": [task.to_dict() for task in db.session.query(VideoProcessingTask).filter(
                VideoProcessingTask.video_id == video_id,
                VideoProcessingTask.task_type.in_(steps_to_run)
            ).all()]
        })
    
    except Exception as e:
        logger.exception(f"启动视频处理流程失败: {str(e)}")
        return error_response(500, f"启动处理流程失败: {str(e)}")

def process_video_workflow(video_id, steps_to_run, app):
    """处理视频的完整工作流程"""
    with app.app_context():
        # 按顺序执行每个步骤
        for step_id in steps_to_run:
            # 查找步骤定义
            step_def = next((s for s in PROCESSING_STEPS if s['id'] == step_id), None)
            if not step_def:
                logger.warning(f"跳过未定义的步骤: {step_id}")
                continue
            
            # 检查依赖项
            if 'depends_on' in step_def:
                # 获取依赖任务状态
                depend_task = db.session.query(VideoProcessingTask).filter(
                    VideoProcessingTask.video_id == video_id,
                    VideoProcessingTask.task_type == step_def['depends_on']
                ).first()
                
                # 如果依赖任务不存在或未完成，跳过此步骤
                if not depend_task or depend_task.status != 'completed':
                    logger.warning(f"依赖步骤 {step_def['depends_on']} 未完成，跳过: {step_id}")
                    continue
            
            # 更新任务状态为处理中
            task = db.session.query(VideoProcessingTask).filter(
                VideoProcessingTask.video_id == video_id,
                VideoProcessingTask.task_type == step_id
            ).first()
            
            if not task:
                logger.warning(f"找不到任务记录: {step_id}")
                continue
            
            try:
                # 更新状态
                task.status = "processing"
                task.started_at = datetime.datetime.utcnow()
                task.progress = 10.0
                db.session.commit()
                
                # 构建API请求URL
                endpoint = step_def['endpoint'].format(video_id=video_id)
                url = f"http://localhost:8000{endpoint}"  # 使用本地地址调用自身API
                
                # 发送请求
                logger.info(f"执行步骤 {step_id}: {url}")
                response = requests.request(
                    method=step_def['method'],
                    url=url,
                    json={}  # 如果需要参数，可以添加
                )
                
                task.progress = 80.0
                db.session.commit()
                
                # 检查响应
                if response.status_code >= 200 and response.status_code < 300:
                    # 成功
                    task.status = "completed"
                    task.progress = 100.0
                    task.completed_at = datetime.datetime.utcnow()
                    db.session.commit()
                    logger.info(f"步骤 {step_id} 完成")
                else:
                    # 失败
                    error_data = response.json()
                    error_msg = error_data.get('message', f"API返回错误: {response.status_code}")
                    raise Exception(error_msg)
            
            except Exception as e:
                logger.exception(f"处理步骤 {step_id} 失败: {str(e)}")
                task.status = "failed"
                task.error = str(e)
                db.session.commit()
                
                # 如果一个步骤失败，后续步骤可能也会失败，但我们继续尝试
                continue

@workflow_api.route('/api/videos/<video_id>/process/status', methods=['GET'])
def get_processing_status(video_id):
    """获取视频处理状态"""
    try:
        # 查询所有任务
        tasks = db.session.query(VideoProcessingTask).filter(
            VideoProcessingTask.video_id == video_id
        ).all()
        
        if not tasks:
            return error_response(404, "未找到处理任务")
        
        # 构建响应
        task_list = [task.to_dict() for task in tasks]
        
        # 计算总进度
        total_progress = sum(task.progress for task in tasks) / len(tasks) if tasks else 0
        
        # 确定总体状态
        if all(task.status == 'completed' for task in tasks):
            total_status = 'completed'
        elif any(task.status == 'failed' for task in tasks):
            total_status = 'failed'
        elif any(task.status == 'processing' for task in tasks):
            total_status = 'processing'
        else:
            total_status = 'pending'
        
        return success_response({
            "video_id": video_id,
            "tasks": task_list,
            "total_progress": total_progress,
            "status": total_status
        })
    
    except Exception as e:
        logger.exception(f"获取处理状态失败: {str(e)}")
        return error_response(500, f"获取状态失败: {str(e)}")

def auto_process_after_upload(file_id):
    """上传完成后自动开始处理视频"""
    try:
        requests.post(
            url=f"http://localhost:8000/api/videos/{file_id}/process",
            json={"steps": ["transcription", "extract", "summary", "assessment", "classify", "report"]}
        )
        logger.info(f"已为视频 {file_id} 启动自动处理")
    except Exception as e:
        logger.exception(f"启动自动处理失败: {str(e)}")