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

# 定义处理步骤 - 添加事实核查和数字人检测
PROCESSING_STEPS = [
    {
        "id": "transcription",
        "name": "视频转录",
        "endpoint": "/api/videos/{video_id}/transcribe",
        "method": "POST",
        "category": "基础处理"
    },
    {
        "id": "fact_check",
        "name": "事实核查",
        "endpoint": "/api/videos/{video_id}/factcheck",
        "method": "POST",
        "depends_on": "transcription",
        "category": "内容分析"
    },
    {
        "id": "digital_human",
        "name": "数字人检测",
        "endpoint": "/api/videos/{video_id}/digital-human/detect",
        "method": "POST",
        "category": "AI检测"
    },
    {
        "id": "extract",
        "name": "信息提取",
        "endpoint": "/api/extract/video/{video_id}",
        "method": "POST",
        "depends_on": "transcription",
        "category": "内容分析"
    },
    {
        "id": "summary",
        "name": "生成摘要",
        "endpoint": "/api/summary/video/{video_id}",
        "method": "POST",
        "depends_on": "extract",
        "category": "内容分析"
    },
    {
        "id": "assessment",
        "name": "内容评估",
        "endpoint": "/api/videos/{video_id}/assess",
        "method": "POST",
        "depends_on": "extract",
        "category": "风险评估"
    },
    {
        "id": "classify",
        "name": "风险分类",
        "endpoint": "/api/videos/{video_id}/classify-risk",
        "method": "POST",
        "depends_on": "assessment",
        "category": "风险评估"
    },
    {
        "id": "report",
        "name": "威胁报告",
        "endpoint": "/api/videos/{video_id}/generate-report",
        "method": "POST",
        "depends_on": "classify",
        "category": "报告生成"
    }
]

# 预定义工作流模板
WORKFLOW_TEMPLATES = {
    "full": {
        "name": "完整分析",
        "description": "包含所有分析步骤的完整工作流",
        "steps": ["transcription", "fact_check", "digital_human", "extract", "summary", "assessment", "classify", "report"]
    },
    "light": {
        "name": "轻量分析", 
        "description": "基础的转录和信息提取",
        "steps": ["transcription", "digital_human","extract", "summary", "assessment", "classify", "report"]
    },
    "content": {
        "name": "内容分析",
        "description": "专注于内容安全和风险评估", 
        "steps": ["transcription", "extract", "summary", "assessment", "classify", "report"]
    },
}

@workflow_api.route('/api/workflow/templates', methods=['GET'])
def get_workflow_templates():
    """获取可用的工作流模板"""
    try:
        return success_response({
            "templates": WORKFLOW_TEMPLATES,
            "steps": [
                {
                    "id": step["id"],
                    "name": step["name"],
                    "category": step["category"],
                    "depends_on": step.get("depends_on")
                }
                for step in PROCESSING_STEPS
            ]
        })
    except Exception as e:
        return error_response(500, f"获取模板失败: {str(e)}")

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
        data = request.get_json() or {}
        
        # 支持模板和自定义步骤
        if 'template' in data:
            template_name = data['template']
            if template_name not in WORKFLOW_TEMPLATES:
                return error_response(400, f"未知的工作流模板: {template_name}")
            steps_to_run = WORKFLOW_TEMPLATES[template_name]['steps']
        else:
            steps_to_run = data.get('steps', [step['id'] for step in PROCESSING_STEPS])
        
        force = data.get('force', False)  # 是否强制重新处理
        
        # 验证步骤
        valid_steps = [step['id'] for step in PROCESSING_STEPS]
        invalid_steps = [s for s in steps_to_run if s not in valid_steps]
        if invalid_steps:
            return error_response(400, f"无效的步骤: {', '.join(invalid_steps)}")
        
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
            "steps": steps_to_run,
            "template": data.get('template'),
            "tasks": [task.to_dict() for task in db.session.query(VideoProcessingTask).filter(
                VideoProcessingTask.video_id == video_id,
                VideoProcessingTask.task_type.in_(steps_to_run)
            ).all()]
        })
    
    except Exception as e:
        logger.exception(f"启动视频处理流程失败: {str(e)}")
        return error_response(500, f"启动处理流程失败: {str(e)}")

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
                
                # 特殊处理数字人检测，需要传递检测类型
                request_data = {}
                if step_id == "digital_human":
                    request_data = {
                        "types": ["face", "body", "overall"],
                        "comprehensive": True
                    }
                
                # 发送请求
                add_processing_log(
                    video_id=video_id,
                    task_id=task.id,
                    task_type=step_id,
                    level="INFO",
                    message=f"调用API: {url}"
                )
                
                # 设置超时时间，数字人检测需要更长时间
                timeout = 1800 if step_id == "digital_human" else 300  # 30分钟 vs 5分钟
                
                response = requests.request(
                    method=step_def['method'],
                    url=url,
                    json=request_data,
                    timeout=timeout
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
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('message', f"API返回错误: {response.status_code}")
                    except:
                        error_msg = f"API返回错误: {response.status_code} - {response.text[:200]}"
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
        
        # 按类别分组任务
        tasks_by_category = {}
        for task in tasks:
            step_def = next((s for s in PROCESSING_STEPS if s['id'] == task.task_type), None)
            category = step_def['category'] if step_def else '其他'
            
            if category not in tasks_by_category:
                tasks_by_category[category] = []
            tasks_by_category[category].append(task.to_dict())
        
        return success_response({
            "video_id": video_id,
            "tasks": task_list,
            "tasks_by_category": tasks_by_category,
            "total_progress": total_progress,
            "status": total_status,
            "completed_tasks": len([t for t in tasks if t.status == 'completed']),
            "failed_tasks": len([t for t in tasks if t.status == 'failed']),
            "total_tasks": len(tasks)
        })
    
    except Exception as e:
        logger.exception(f"获取处理状态失败: {str(e)}")
        return error_response(500, f"获取状态失败: {str(e)}")

@workflow_api.route('/api/videos/<video_id>/process/cancel', methods=['POST'])
def cancel_processing(video_id):
    """取消视频处理"""
    try:
        # 更新所有处理中的任务为取消状态
        processing_tasks = db.session.query(VideoProcessingTask).filter(
            VideoProcessingTask.video_id == video_id,
            VideoProcessingTask.status == 'processing'
        ).all()
        
        for task in processing_tasks:
            task.status = 'cancelled'
            task.error = '用户取消'
        
        db.session.commit()
        
        add_processing_log(
            video_id=video_id,
            level="INFO",
            message=f"用户取消了 {len(processing_tasks)} 个处理任务"
        )
        
        return success_response({
            "message": f"已取消 {len(processing_tasks)} 个处理任务",
            "cancelled_tasks": len(processing_tasks)
        })
        
    except Exception as e:
        logger.exception(f"取消处理失败: {str(e)}")
        return error_response(500, f"取消失败: {str(e)}")

def start_video_workflow(video_id, template="content", custom_steps=None, app_context=None):
    """
    统一的视频工作流启动函数
    
    Args:
        video_id: 视频ID
        template: 工作流模板名称 (full, light, content, ai_detection, fact_checking)
        custom_steps: 自定义步骤列表，如果提供则忽略template
        app_context: Flask应用上下文，用于异步调用
    """
    try:
        # 确定要执行的步骤
        if custom_steps:
            steps_to_run = custom_steps
        elif template in WORKFLOW_TEMPLATES:
            steps_to_run = WORKFLOW_TEMPLATES[template]['steps']
        else:
            logger.warning(f"未知的工作流模板: {template}，使用默认content模板")
            steps_to_run = WORKFLOW_TEMPLATES['content']['steps']
        
        # 如果提供了app_context，使用异步方式启动
        if app_context:
            thread = threading.Thread(
                target=process_video_workflow,
                args=(video_id, steps_to_run, app_context)
            )
            thread.daemon = True
            thread.start()
            logger.info(f"异步启动视频 {video_id} 的 {template} 工作流")
        else:
            # 同步启动工作流（直接调用API）
            response = requests.post(
                url=f"http://localhost:8000/api/videos/{video_id}/process",
                json={"template": template, "steps": steps_to_run}
            )
            logger.info(f"同步启动视频 {video_id} 的 {template} 工作流：{response.status_code}")
            return response
            
    except Exception as e:
        logger.exception(f"启动工作流失败: {str(e)}")
        return None
def auto_process_after_upload(file_id):
    """上传完成后自动开始处理视频"""
    try:
        requests.post(
            url=f"http://localhost:8000/api/videos/{file_id}/process",
            json={"template": "full"}  # 使用完整分析模板
        )
        logger.info(f"已为视频 {file_id} 启动自动处理")
    except Exception as e:
        logger.exception(f"启动自动处理失败: {str(e)}")