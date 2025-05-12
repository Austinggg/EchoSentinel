from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
import hashlib
from datetime import datetime, timedelta
# 在现有的导入中添加
import requests
import re
import time
from utils.database import DouyinVideo, db, UserProfile, UserAnalysisTask
import json
account_api = Blueprint('account_api', __name__)

# 现有的add_account函数处理已经基本完善
# 可能需要更新以下这个函数以确保它接收正确的字段

@account_api.route('/api/account/add', methods=['POST'])
def add_account():
    """添加用户账号并创建分析任务"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"code": 400, "message": "无效的请求数据"}), 400
        
        # 必需字段校验
        required_fields = ['platform', 'platform_user_id', 'nickname']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"code": 400, "message": f"缺少必需字段: {field}"}), 400
        
        # 检查用户是否已存在
        sec_uid = data.get('sec_uid', data.get('platform_user_id'))
        hash_sec_uid = hashlib.md5(sec_uid.encode()).hexdigest()
        
        existing_profile = UserProfile.query.filter_by(hash_sec_uid=hash_sec_uid).first()
        
        # 如果用户不存在，则创建新用户
        if not existing_profile:
            # 创建用户资料
            user_profile = UserProfile(
                sec_uid=sec_uid,
                hash_sec_uid=hash_sec_uid,
                nickname=data.get('nickname'),
                gender=data.get('gender', ''),
                city=data.get('city', ''),
                province=data.get('province', ''),
                country=data.get('country', ''),
                aweme_count=data.get('aweme_count', 0),
                follower_count=data.get('follower_count', 0),
                following_count=data.get('following_count', 0),
                total_favorited=data.get('total_favorited', 0),
                favoriting_count=data.get('favoriting_count', 0),
                user_age=data.get('user_age', 0),
                ip_location=data.get('location', ''),
                avatar_medium=data.get('avatar', '')
            )
            db.session.add(user_profile)
            db.session.flush()  # 获取ID但不提交
            profile_id = user_profile.id
        else:
            # 更新现有用户信息
            profile_id = existing_profile.id
            existing_profile.nickname = data.get('nickname', existing_profile.nickname)
            existing_profile.aweme_count = data.get('aweme_count', existing_profile.aweme_count)
            existing_profile.follower_count = data.get('follower_count', existing_profile.follower_count)
            existing_profile.following_count = data.get('following_count', existing_profile.following_count)
            existing_profile.total_favorited = data.get('total_favorited', existing_profile.total_favorited)
            existing_profile.avatar_medium = data.get('avatar', existing_profile.avatar_medium)
            existing_profile.ip_location = data.get('location', existing_profile.ip_location)
            existing_profile.user_age = data.get('user_age', existing_profile.user_age)
        
        # 创建分析任务
        analysis_task = UserAnalysisTask(
            platform=data.get('platform'),
            platform_user_id=data.get('platform_user_id'),
            nickname=data.get('nickname'),
            avatar=data.get('avatar', ''),
            user_profile_id=profile_id,
            status='pending',
            progress=0,
            analysis_type='full',
            max_videos=50,  # 默认最多分析50个视频
            created_at=datetime.now()
        )
        
        db.session.add(analysis_task)
        db.session.commit()
        
        return jsonify({
            "code": 200,
            "message": "账号添加成功",
            "data": {
                "user_id": profile_id,
                "task_id": analysis_task.id
            }
        })
        
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"code": 500, "message": f"数据库错误: {str(e)}"}), 500
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "message": f"服务器错误: {str(e)}"}), 500


# 添加在其他函数后
@account_api.route('/api/account/<int:profile_id>/fetch_videos', methods=['POST'])
def fetch_account_videos(profile_id):
    """获取抖音用户的视频并保存到数据库"""
    try:
        # 获取用户资料
        user_profile = UserProfile.query.get(profile_id)
        if not user_profile:
            return jsonify({"code": 404, "message": "用户不存在"}), 404
        
        # 获取请求参数
        data = request.get_json() or {}
        max_videos = data.get('max_videos', 20)  # 默认最多获取20个视频
        
        # 获取视频列表
        videos_added = fetch_and_save_videos(user_profile.sec_uid, profile_id, max_videos)
        
        return jsonify({
            "code": 200,
            "message": f"成功获取并保存了 {videos_added} 个视频",
            "data": {
                "videos_added": videos_added
            }
        })
        
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取视频失败: {str(e)}"}), 500
# 在现有函数后添加以下代码
@account_api.route('/api/account/videos/<string:aweme_id>/analyze', methods=['POST'])
def analyze_douyin_video(aweme_id):
    """分析抖音视频"""
    try:
        # 获取视频信息
        video = DouyinVideo.query.filter_by(aweme_id=aweme_id).first()
        
        if not video:
            return jsonify({"code": 404, "message": "视频不存在"}), 404
        
        # 如果视频已有分析结果，直接返回
        if video.video_file_id:
            from utils.database import ContentAnalysis
            
            content_analysis = ContentAnalysis.query.filter_by(
                video_id=video.video_file_id
            ).first()
            
            if content_analysis and content_analysis.risk_level:
                return jsonify({
                    "code": 200,
                    "message": "视频已分析过",
                    "data": {
                        "video_id": video.video_file_id,
                        "risk_level": content_analysis.risk_level,
                        "risk_probability": content_analysis.risk_probability,
                        "summary": content_analysis.summary
                    }
                })
        
        # 获取视频URL
        video_url = video.share_url
        
        if not video_url:
            return jsonify({"code": 400, "message": "视频链接不可用，无法下载"}), 400
        
        # 调用下载并分析API
        download_analyze_response = requests.get(
            f"http://localhost:8000/api/download_and_analyze",
            params={
                "url": video_url,
                "prefix": "false",
                "with_watermark": "false"
            }
        )
        
        # 检查HTTP状态码
        if download_analyze_response.status_code != 200:
            return jsonify({
                "code": 500, 
                "message": f"视频下载分析失败: HTTP {download_analyze_response.status_code}"
            }), 500
        
        # 安全解析JSON
        try:
            response_data = download_analyze_response.json()
        except Exception as e:
            return jsonify({
                "code": 500,
                "message": f"解析响应失败: {str(e)}"
            }), 500
        
        # 检查API响应状态
        if response_data.get("code") != 200:
            return jsonify({
                "code": 500, 
                "message": f"视频下载分析失败: {response_data.get('message')}"
            }), 500
        
        # 获取文件ID
        file_id = response_data.get("data", {}).get("fileId")
        if not file_id:
            return jsonify({"code": 500, "message": "未能获取视频ID"}), 500
            
        # 更新视频关联
        video.video_file_id = file_id
        db.session.commit()
        
        # 返回成功响应
        return jsonify({
            "code": 200,
            "message": "分析任务已启动",
            "data": {
                "video_id": file_id,
                "status": "processing"
            }
        })
            
    except Exception as e:
        print(f"分析视频失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"分析视频失败: {str(e)}"}), 500
# 修改 get_video_processing_details 函数，添加最近的日志信息

@account_api.route('/api/account/videos/<string:aweme_id>/processing-details', methods=['GET'])
def get_video_processing_details(aweme_id):
    """获取视频处理的详细进度信息"""
    try:
        # 获取视频信息
        video = DouyinVideo.query.filter_by(aweme_id=aweme_id).first()
        
        if not video:
            return jsonify({"code": 404, "message": "视频不存在"}), 404
            
        if not video.video_file_id:
            return jsonify({
                "code": 200,
                "data": {
                    "status": "not_downloaded",
                    "message": "视频尚未下载",
                    "tasks": []
                }
            })
        
        # 查询处理任务
        from utils.database import VideoProcessingTask, VideoFile, ProcessingLog
        
        # 获取视频文件信息
        video_file = VideoFile.query.get(video.video_file_id)
        if not video_file:
            return jsonify({"code": 404, "message": "视频文件不存在"}), 404
            
        # 获取处理任务列表
        tasks = VideoProcessingTask.query.filter_by(video_id=video.video_file_id).all()
        tasks_data = []
        
        # 为每个任务添加最近的日志
        for task in tasks:
            task_dict = task.to_dict()
            # 获取该任务最近的10条日志
            recent_logs = ProcessingLog.query.filter_by(
                video_id=video.video_file_id, 
                task_id=task.id
            ).order_by(ProcessingLog.created_at.desc()).limit(10).all()
            
            # 添加日志到任务数据中
            task_dict["logs"] = [log.to_dict() for log in recent_logs]
            tasks_data.append(task_dict)
        
        # 确定总体进度
        overall_status = "completed"
        if any(task.status == "failed" for task in tasks):
            overall_status = "failed"
        elif any(task.status == "processing" for task in tasks):
            overall_status = "processing"
        elif any(task.status == "pending" for task in tasks):
            overall_status = "pending"
            
        # 计算总体进度百分比
        if tasks:
            overall_progress = sum(task.progress for task in tasks) / len(tasks)
        else:
            overall_progress = 0
        
        # 计算完成时间（如果已完成）
        completed_time = None
        if overall_status == "completed" and tasks:
            completed_tasks = [t for t in tasks if t.completed_at]
            if completed_tasks:
                latest_completion = max(t.completed_at for t in completed_tasks)
                completed_time = latest_completion.isoformat()
            
        return jsonify({
            "code": 200,
            "data": {
                "video_id": video.video_file_id,
                "aweme_id": video.aweme_id,
                "desc": video.desc,
                "cover_url": video.cover_url,
                "status": overall_status,
                "progress": overall_progress,
                "source_type": "upload" if not video.share_url else "download",
                "download_time": video_file.upload_time.isoformat() if video_file.upload_time else None,
                "completed_time": completed_time,
                "tasks": tasks_data
            }
        })
        
    except Exception as e:
        print(f"获取视频处理详情失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"获取视频处理详情失败: {str(e)}"}), 500
@account_api.route('/api/account/videos/<string:aweme_id>/info', methods=['GET'])
def get_video_info(aweme_id):
    """获取视频基本信息，包括所属用户ID"""
    try:
        # 获取视频信息
        video = DouyinVideo.query.filter_by(aweme_id=aweme_id).first()
        
        if not video:
            return jsonify({"code": 404, "message": "视频不存在"}), 404
            
        # 返回视频信息，包括所属用户ID
        return jsonify({
            "code": 200,
            "data": {
                "aweme_id": video.aweme_id,
                "desc": video.desc,
                "cover_url": video.cover_url,
                "profile_id": video.user_profile_id,
                "video_file_id": video.video_file_id
            }
        })
        
    except Exception as e:
        print(f"获取视频信息失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"获取视频信息失败: {str(e)}"}), 500
@account_api.route('/api/analysis/tasks', methods=['GET'])
def get_analysis_tasks():
    """获取用户分析任务列表"""
    try:
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search = request.args.get('search', '')
        platform = request.args.get('platform')
        status = request.args.get('status')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 构建查询
        query = UserAnalysisTask.query
        
        # 应用过滤条件
        if search:
            query = query.filter(UserAnalysisTask.nickname.like(f"%{search}%"))
        if platform:
            query = query.filter(UserAnalysisTask.platform == platform)
        if status:
            query = query.filter(UserAnalysisTask.status == status)
        
        # 应用日期范围过滤
        if start_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(UserAnalysisTask.created_at >= start_date_obj)
        if end_date:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            end_date_obj = end_date_obj + timedelta(days=1)  # 包括结束日期的全天
            query = query.filter(UserAnalysisTask.created_at < end_date_obj)
        
        # 应用排序
        if hasattr(UserAnalysisTask, sort_by):
            sort_column = getattr(UserAnalysisTask, sort_by)
            if sort_order == 'desc':
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column)
        else:
            # 默认按创建时间降序排列
            query = query.order_by(UserAnalysisTask.created_at.desc())
        
        # 执行分页查询
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # 准备响应数据
        tasks = []
        for task in pagination.items:
            task_dict = {
                "id": task.id,
                "platform": task.platform,
                "platform_user_id": task.platform_user_id,
                "user_profile_id": task.user_profile_id,
                "nickname": task.nickname,
                "avatar": task.avatar,
                "status": task.status,
                "progress": task.progress,
                "error": task.error,
                "analysis_type": task.analysis_type,
                "result_summary": task.result_summary,
                "risk_level": task.risk_level,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            tasks.append(task_dict)
        
        return jsonify({
            "code": 200,
            "message": "获取成功",
            "data": {
                "tasks": tasks,
                "total": pagination.total,
                "current_page": page,
                "per_page": per_page,
                "total_pages": pagination.pages
            }
        })
        
    except Exception as e:
        print(f"获取任务列表失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"获取任务列表失败: {str(e)}"}), 500
@account_api.route('/api/account/by-secuid/<string:sec_uid>', methods=['GET'])
def get_account_by_secuid(sec_uid):
    """根据sec_uid获取用户信息"""
    try:
        # 计算hash用于查询
        hash_sec_uid = hashlib.md5(sec_uid.encode()).hexdigest()
        
        # 查询用户
        user_profile = UserProfile.query.filter_by(hash_sec_uid=hash_sec_uid).first()
        
        if not user_profile:
            return jsonify({"code": 404, "message": "用户不存在"}), 404
        
        # 返回用户数据
        return jsonify({
            "code": 200,
            "data": {
                "id": user_profile.id,
                "nickname": user_profile.nickname,
                "sec_uid": user_profile.sec_uid,
                "avatar": user_profile.avatar_medium,
                "follower_count": user_profile.follower_count,
                "following_count": user_profile.following_count,
                "aweme_count": user_profile.aweme_count,
                "total_favorited": user_profile.total_favorited,
                "signature": user_profile.signature,
                "gender": user_profile.gender,
                "ip_location": user_profile.ip_location,
                "user_age": user_profile.user_age
            }
        })
        
    except Exception as e:
        print(f"获取用户信息失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"获取用户信息失败: {str(e)}"}), 500
@account_api.route('/api/account/<int:profile_id>/videos', methods=['GET'])
def get_account_videos(profile_id):
    """获取用户的视频列表"""
    try:
        # 检查用户是否存在
        user_profile = UserProfile.query.get(profile_id)
        if not user_profile:
            return jsonify({"code": 404, "message": "用户不存在"}), 404
        
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        sort_by = request.args.get('sort_by', 'create_time')
        sort_order = request.args.get('sort_order', 'desc')
        search = request.args.get('search', '')
        
        # 构建查询
        query = DouyinVideo.query.filter_by(user_profile_id=profile_id)
        
        # 应用搜索过滤
        if search:
            query = query.filter(
                db.or_(
                    DouyinVideo.desc.ilike(f'%{search}%'),
                    DouyinVideo.tags.ilike(f'%{search}%')
                )
            )
        
        # 应用排序
        if hasattr(DouyinVideo, sort_by):
            sort_column = getattr(DouyinVideo, sort_by)
            if sort_order.lower() == 'desc':
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column)
        else:
            # 默认按创建时间降序排序
            query = query.order_by(DouyinVideo.create_time.desc())
        
        # 执行分页查询
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # 格式化视频数据
        videos = [video.to_dict() for video in pagination.items]
        
        # 返回结果
        return jsonify({
            "code": 200,
            "message": "获取成功",
            "data": {
                "videos": videos,
                "total": pagination.total,
                "current_page": page,
                "per_page": per_page,
                "total_pages": pagination.pages
            }
        })
        
    except Exception as e:
        print(f"获取视频列表失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"获取视频列表失败: {str(e)}"}), 500

    
@account_api.route('/api/account/videos/<string:aweme_id>/analysis-status', methods=['GET'])
def get_douyin_video_analysis_status(aweme_id):
    """获取视频分析状态"""
    try:
        # 获取视频信息
        video = DouyinVideo.query.filter_by(aweme_id=aweme_id).first()
        
        if not video:
            return jsonify({"code": 404, "message": "视频不存在"}), 404
            
        if not video.video_file_id:
            return jsonify({
                "code": 200,
                "data": {
                    "status": "not_downloaded",
                    "message": "视频尚未下载"
                }
            })
        
        # 查询分析状态
        from utils.database import VideoProcessingTask, ContentAnalysis
        
        content_analysis = ContentAnalysis.query.filter_by(
            video_id=video.video_file_id
        ).first()
        
        processing_tasks = VideoProcessingTask.query.filter_by(
            video_id=video.video_file_id
        ).all()
        
        # 检查分析是否已完成
        if content_analysis and content_analysis.risk_level:
            return jsonify({
                "code": 200,
                "data": {
                    "status": "completed",
                    "risk_level": content_analysis.risk_level,
                    "risk_probability": content_analysis.risk_probability,
                    "message": "分析已完成",
                    "video_id": video.video_file_id
                }
            })
        
        # 检查是否正在处理
        active_tasks = [t for t in processing_tasks if t.status in ('pending', 'processing')]
        if active_tasks:
            # 计算总体进度
            total_progress = sum(t.progress for t in active_tasks) / len(active_tasks)
            return jsonify({
                "code": 200,
                "data": {
                    "status": "processing",
                    "message": "视频分析中",
                    "progress": total_progress,
                    "tasks": [{"type": t.task_type, "status": t.status, "progress": t.progress} for t in active_tasks],
                    "video_id": video.video_file_id
                }
            })
        
        # 没有活动任务，但也没完成分析，可能是任务失败或尚未开始
        failed_tasks = [t for t in processing_tasks if t.status == 'failed']
        if failed_tasks:
            return jsonify({
                "code": 200,
                "data": {
                    "status": "failed",
                    "message": "分析失败",
                    "error": failed_tasks[0].error,
                    "video_id": video.video_file_id
                }
            })
        
        # 状态不明确
        return jsonify({
            "code": 200,
            "data": {
                "status": "unknown",
                "message": "未知状态，可能需要重新开始分析",
                "video_id": video.video_file_id
            }
        })
        
    except Exception as e:
        print(f"获取分析状态失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"获取分析状态失败: {str(e)}"}), 500

@account_api.route('/api/account/<int:profile_id>/stats', methods=['GET'])
def get_account_stats(profile_id):
    """获取用户视频分析统计数据"""
    try:
        # 检查用户是否存在
        user_profile = UserProfile.query.get(profile_id)
        if not user_profile:
            return jsonify({"code": 404, "message": "用户不存在"}), 404
        
        # 获取所有视频
        videos = DouyinVideo.query.filter_by(user_profile_id=profile_id).all()
        
        total_videos = len(videos)
        if total_videos == 0:
            return jsonify({
                "code": 200, 
                "message": "暂无视频数据",
                "data": {
                    "total_videos": 0,
                    "analyzed_videos": 0,
                    "pending_videos": 0,
                    "risk_distribution": []
                }
            })
        
        # 统计数据
        videos_with_file_id = 0
        analyzed_videos = 0
        risk_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
        
        # 查询分析结果
        from utils.database import VideoFile, ContentAnalysis
        
        for video in videos:
            if video.video_file_id:
                videos_with_file_id += 1
                
                # 查询内容分析结果
                content_analysis = ContentAnalysis.query.filter_by(video_id=video.video_file_id).first()
                if content_analysis and content_analysis.risk_level:
                    analyzed_videos += 1
                    risk_level = content_analysis.risk_level.lower()
                    
                    # 统计风险级别
                    if risk_level in risk_counts:
                        risk_counts[risk_level] += 1
                    else:
                        risk_counts["unknown"] += 1
        
        # 计算待分析视频
        pending_videos = total_videos - analyzed_videos
        
        # 格式化为图表所需数据格式
        risk_distribution = [
            {"value": risk_counts["low"], "name": "低风险"},
            {"value": risk_counts["medium"], "name": "中风险"},
            {"value": risk_counts["high"], "name": "高风险"},
            {"value": risk_counts["unknown"], "name": "未知风险"}
        ]
        
        # 分析状态分布
        analysis_status = [
            {"value": analyzed_videos, "name": "已分析"},
            {"value": videos_with_file_id - analyzed_videos, "name": "分析中"},
            {"value": total_videos - videos_with_file_id, "name": "未下载"}
        ]
        
        return jsonify({
            "code": 200,
            "message": "获取成功",
            "data": {
                "total_videos": total_videos,
                "analyzed_videos": analyzed_videos,
                "pending_videos": pending_videos,
                "risk_distribution": risk_distribution,
                "analysis_status": analysis_status
            }
        })
        
    except Exception as e:
        print(f"获取统计数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"获取统计数据失败: {str(e)}"}), 500
def fetch_and_save_videos(sec_uid, user_profile_id, max_count=20):
    """获取并保存用户视频"""
    try:
        # 变量初始化
        videos_added = 0
        max_cursor = 0
        has_more = True
        retry_count = 0
        docker_service_host = "http://localhost:80"
        
        print(f"准备获取用户 {sec_uid} 的视频，最大数量: {max_count}")
        
        # 循环获取视频直到达到最大数量或没有更多视频
        while videos_added < max_count and has_more and retry_count < 5:
            # 构建API请求
            url = f"{docker_service_host}/api/douyin/web/fetch_user_post_videos"
            params = {
                "sec_user_id": sec_uid,
                "max_cursor": max_cursor,
                "count": 50  # 固定请求20个
            }
            
            print(f"发送请求: {url}, 参数: {params}, 已添加: {videos_added}/{max_count}")
            
            # 发送请求
            response = requests.get(url, params=params)
            
            # 检查响应状态
            if response.status_code != 200:
                print(f"API请求失败: HTTP {response.status_code}")
                retry_count += 1
                time.sleep(2)
                continue
            
            # 解析响应数据
            data = response.json()
            if data.get("code") != 200:
                print(f"API返回错误: {data.get('message')}")
                retry_count += 1
                time.sleep(2)
                continue
            
            # 获取视频列表
            aweme_list = data.get("data", {}).get("aweme_list", [])
            print(f"本次获取到 {len(aweme_list)} 个视频")
            
            if not aweme_list:
                print("返回的视频列表为空，停止获取")
                break
            
            # 记录每个视频的ID用于调试
            video_ids = [video.get("aweme_id") for video in aweme_list]
            print(f"视频ID列表: {video_ids}")
            
            # 处理视频数据
            videos_in_batch = 0
            for video_data in aweme_list:
                result = process_video(video_data, user_profile_id)
                if result:
                    videos_added += 1
                    videos_in_batch += 1
                
                # 如果达到最大数量则停止
                if videos_added >= max_count:
                    print(f"已达到最大数量 {max_count}，停止获取")
                    break
            
            print(f"本批次成功处理 {videos_in_batch} 个视频，总计: {videos_added}")
            
            # 更新分页信息 - 即使没有更多数据也更新cursor
            old_cursor = max_cursor
            max_cursor = data.get("data", {}).get("max_cursor", 0)
            has_more = data.get("data", {}).get("has_more") == 1
            
            print(f"分页信息: 原cursor={old_cursor}, 新cursor={max_cursor}, has_more={has_more}")
            
            # 避免请求过快，增加延迟
            time.sleep(3)
            
            # 如果cursor没有变化且还有更多数据，可能是API问题，尝试手动增加
            if old_cursor == max_cursor and has_more:
                print("警告: cursor未变化，手动调整")
                max_cursor += 1
                
        print(f"获取完成，共添加 {videos_added} 个视频")
        return videos_added
        
    except Exception as e:
        print(f"获取视频异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return videos_added
def process_video(video_data, user_profile_id):
    """处理单个视频数据并保存到数据库"""
    try:
        # 提取视频ID
        aweme_id = video_data.get("aweme_id")
        if not aweme_id:
            print("视频缺少ID")
            return None
        
        # 检查视频是否已存在
        existing_video = DouyinVideo.query.filter_by(aweme_id=aweme_id).first()
        if existing_video:
            # 更新统计数据
            update_video_stats(existing_video, video_data)
            return existing_video.id
        
        # 提取创建时间
        create_time = datetime.fromtimestamp(video_data.get("create_time", 0))
        
        # 提取封面URL
        cover_url = ""
        if video_data.get("video") and video_data["video"].get("cover"):
            cover_urls = video_data["video"]["cover"].get("url_list", [])
            if cover_urls:
                cover_url = cover_urls[0]
        
        # 提取分享URL
        share_url = ""
        if video_data.get("share_info"):
            share_url = video_data["share_info"].get("share_url", "")
        
        # 提取视频时长
        duration = 0
        if video_data.get("video") and video_data["video"].get("duration"):
            duration = int(video_data["video"]["duration"]) // 1000  # 毫秒转秒
        
        # 提取统计数据
        statistics = video_data.get("statistics", {})
        
        # 提取标签
        tags = extract_tags(video_data)
        
        # 创建新视频记录
        new_video = DouyinVideo(
            aweme_id=aweme_id,
            user_profile_id=user_profile_id,
            desc=video_data.get("desc", ""),
            create_time=create_time,
            cover_url=cover_url,
            share_url=share_url,
            media_type=video_data.get("aweme_type", 0),
            video_duration=duration,
            is_top=bool(video_data.get("is_top", 0)),
            digg_count=statistics.get("digg_count", 0),
            comment_count=statistics.get("comment_count", 0),
            collect_count=statistics.get("collect_count", 0),
            share_count=statistics.get("share_count", 0),
            play_count=statistics.get("play_count", 0),
            tags=tags,
            fetched_at=datetime.utcnow()
        )
        
        db.session.add(new_video)
        db.session.commit()
        
        return new_video.id
        
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"保存视频数据时出错: {str(e)}")
        return None
    except Exception as e:
        print(f"处理视频数据时出错: {str(e)}")
        return None
def update_video_stats(video, video_data):
    """更新已存在视频的统计数据"""
    try:
        statistics = video_data.get("statistics", {})
        video.digg_count = statistics.get("digg_count", video.digg_count)
        video.comment_count = statistics.get("comment_count", video.comment_count)
        video.collect_count = statistics.get("collect_count", video.collect_count)
        video.share_count = statistics.get("share_count", video.share_count)
        video.play_count = statistics.get("play_count", video.play_count)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"更新视频统计数据时出错: {str(e)}")
        return False
def extract_tags(video_data):
    """从视频数据中提取标签"""
    tags = []
    
    # 从text_extra中提取标签
    text_extra = video_data.get("text_extra", [])
    for item in text_extra:
        if item.get("hashtag_name"):
            tags.append(item["hashtag_name"])
    
    # 从描述中提取标签
    desc = video_data.get("desc", "")
    if desc:
        hashtags = re.findall(r'#(\w+)', desc)
        tags.extend(hashtags)
    
    # 去重并限制标签数量
    return ",".join(list(set(tags))[:20])