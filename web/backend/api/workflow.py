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
@workflow_api.route('/api/videos/<video_id>/logs', methods=['GET'])
def get_processing_logs(video_id):
    """获取视频处理日志"""
    try:
        # 获取过滤参数
        task_type = request.args.get('task_type')
        level = request.args.get('level')
        limit = int(request.args.get('limit', 100))
        
        # 构建查询
        from utils.database import ProcessingLog
        query = db.session.query(ProcessingLog).filter(ProcessingLog.video_id == video_id)
        
        if task_type:
            query = query.filter(ProcessingLog.task_type == task_type)
        
        if level:
            query = query.filter(ProcessingLog.level == level)
            
        # 按时间排序，最新的先返回
        logs = query.order_by(ProcessingLog.created_at.desc()).limit(limit).all()
        
        # 返回结果
        return success_response({
            "video_id": video_id,
            "logs": [log.to_dict() for log in logs]
        })
        
    except Exception as e:
        logger.exception(f"获取处理日志失败: {str(e)}")
        return error_response(500, f"获取日志失败: {str(e)}")
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

# 修改 process_video_workflow 函数
def add_processing_log(video_id, task_id=None, task_type=None, level="INFO", message=""):
    """添加处理日志到数据库"""
    from utils.database import ProcessingLog, db
    
    try:
        log = ProcessingLog(
            video_id=video_id,
            task_id=task_id,
            task_type=task_type,
            level=level,
            message=message
        )
        db.session.add(log)
        db.session.commit()
        
        # 同时输出到终端
        logger.log(getattr(logging, level), message)
    except Exception as e:
        logger.error(f"记录日志失败: {str(e)}")

def process_video_workflow(video_id, steps_to_run, app):
    """处理视频的完整工作流程"""
    with app.app_context():
        # 记录工作流开始的日志
        add_processing_log(
            video_id=video_id,
            level="INFO",
            message=f"开始视频处理工作流，包含任务: {', '.join(steps_to_run)}"
        )
        
        # 按顺序执行每个步骤
        for step_id in steps_to_run:
            # 查找步骤定义
            step_def = next((s for s in PROCESSING_STEPS if s['id'] == step_id), None)
            if not step_def:
                add_processing_log(
                    video_id=video_id, 
                    level="WARNING",
                    message=f"跳过未定义的步骤: {step_id}"
                )
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
                    add_processing_log(
                        video_id=video_id,
                        level="WARNING",
                        message=f"依赖步骤 {step_def['depends_on']} 未完成，跳过: {step_id}"
                    )
                    continue
            
            # 更新任务状态为处理中
            task = db.session.query(VideoProcessingTask).filter(
                VideoProcessingTask.video_id == video_id,
                VideoProcessingTask.task_type == step_id
            ).first()
            
            if not task:
                add_processing_log(
                    video_id=video_id,
                    level="WARNING",
                    message=f"找不到任务记录: {step_id}"
                )
                continue
            
            try:
                # 更新状态
                task.status = "processing"
                task.started_at = datetime.datetime.utcnow()
                task.progress = 10.0
                db.session.commit()
                
                # 记录开始日志
                add_processing_log(
                    video_id=video_id,
                    task_id=task.id,
                    task_type=step_id,
                    level="INFO",
                    message=f"开始执行 {step_def['name']} 任务"
                )
                
                # 构建API请求URL
                endpoint = step_def['endpoint'].format(video_id=video_id)
                url = f"http://localhost:8000{endpoint}"
                
                # 发送请求
                add_processing_log(
                    video_id=video_id,
                    task_id=task.id,
                    task_type=step_id,
                    level="INFO",
                    message=f"调用API: {url}"
                )
                
                response = requests.request(
                    method=step_def['method'],
                    url=url,
                    json={}
                )
                
                task.progress = 80.0
                db.session.commit()
                
                # 记录响应状态
                add_processing_log(
                    video_id=video_id,
                    task_id=task.id,
                    task_type=step_id,
                    level="INFO",
                    message=f"API响应状态: {response.status_code}"
                )
                
                # 检查响应
                if response.status_code >= 200 and response.status_code < 300:
                    # 成功
                    task.status = "completed"
                    task.progress = 100.0
                    task.completed_at = datetime.datetime.utcnow()
                    db.session.commit()
                    
                    add_processing_log(
                        video_id=video_id,
                        task_id=task.id,
                        task_type=step_id,
                        level="INFO",
                        message=f"{step_def['name']} 任务完成"
                    )
                else:
                    # 失败
                    error_data = response.json()
                    error_msg = error_data.get('message', f"API返回错误: {response.status_code}")
                    raise Exception(error_msg)
            
            except Exception as e:
                add_processing_log(
                    video_id=video_id,
                    task_id=task.id,
                    task_type=step_id,
                    level="ERROR",
                    message=f"处理失败: {str(e)}"
                )
                
                task.status = "failed"
                task.error = str(e)
                db.session.commit()
                continue
                
        # 记录工作流结束的日志
        add_processing_log(
            video_id=video_id,
            level="INFO",
            message="视频处理工作流完成"
        )

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