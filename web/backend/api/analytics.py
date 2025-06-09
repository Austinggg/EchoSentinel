import logging
from datetime import datetime, timedelta
from flask import Blueprint, jsonify
from sqlalchemy import func, and_, distinct, cast, Date,case

from utils.database import (
    UserAnalysisTask, VideoTranscript, db, VideoFile, ContentAnalysis, VideoProcessingTask, 
    UserProfile, DouyinVideo, ProcessingLog
)
from utils.HttpResponse import success_response, error_response

logger = logging.getLogger(__name__)
analytics_api = Blueprint('analytics', __name__)

@analytics_api.route('/api/analytics/overview', methods=['GET'])
def get_analytics_overview():
    """获取概览数据统计"""
    try:
        # 总视频数
        total_videos = db.session.query(VideoFile).count()
        
        # 风险等级分布
        risk_stats = db.session.query(
            VideoFile.risk_level, 
            func.count(VideoFile.id).label('count')
        ).group_by(VideoFile.risk_level).all()
        
        risk_distribution = {
            'low': 0, 
            'medium': 0, 
            'high': 0, 
            'processing': 0
        }
        
        for risk_level, count in risk_stats:
            if risk_level in risk_distribution:
                risk_distribution[risk_level] = count
        
        # 最近7天上传的视频数
        week_ago = datetime.now() - timedelta(days=7)
        recent_videos = db.session.query(VideoFile).filter(
            VideoFile.upload_time >= week_ago
        ).count()
        
        # 处理任务数据
        total_tasks = db.session.query(VideoProcessingTask).count()
        completed_tasks = db.session.query(VideoProcessingTask).filter(
            VideoProcessingTask.status == 'completed'
        ).count()
        failed_tasks = db.session.query(VideoProcessingTask).filter(
            VideoProcessingTask.status == 'failed'
        ).count()
        
        completion_rate = round(completed_tasks / total_tasks * 100, 2) if total_tasks > 0 else 0
        
        # 修改用户数据统计
        # 1. 实际分析的用户数量（去重）
        analysed_users = db.session.query(func.count(distinct(UserAnalysisTask.platform_user_id))).scalar() or 0
        
        # 2. 高风险用户数量
        high_risk_users = db.session.query(func.count(distinct(UserAnalysisTask.platform_user_id)))\
            .filter(UserAnalysisTask.risk_level == 'high').scalar() or 0
        
        # 3. 数字人用户数量（概率>0.7视为数字人）
        digital_human_users = db.session.query(func.count(distinct(UserAnalysisTask.platform_user_id)))\
            .filter(UserAnalysisTask.digital_human_probability >= 0.7).scalar() or 0
        
        return success_response({
            'total_videos': total_videos,
            'risk_distribution': risk_distribution,
            'recent_videos': recent_videos,
            'task_stats': {
                'total': total_tasks,
                'completed': completed_tasks,
                'failed': failed_tasks,
                'completion_rate': completion_rate
            },
            # 更改返回数据结构，添加更详细的用户统计
            'user_stats': {
                'analysed_users': analysed_users,
                'high_risk_users': high_risk_users,
                'digital_human_users': digital_human_users
            },
            # 保留原字段以兼容
            'total_users': analysed_users
        })
        
    except Exception as e:
        logger.exception(f"获取统计概览失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")

@analytics_api.route('/api/analytics/trends', methods=['GET'])
def get_analytics_trends():
    """获取趋势数据"""
    try:
        # 最近14天的数据
        days = 14
        start_date = datetime.now() - timedelta(days=days)
        
        # 按日期分组查询上传视频数
        daily_uploads = db.session.query(
            cast(VideoFile.upload_time, Date).label('date'),
            func.count(VideoFile.id).label('count')
        ).filter(
            VideoFile.upload_time >= start_date
        ).group_by(cast(VideoFile.upload_time, Date)).all()
        
        # 获取每日风险视频数量
        high_risk_daily = db.session.query(
            cast(VideoFile.upload_time, Date).label('date'),
            func.count(VideoFile.id).label('count')
        ).filter(
            VideoFile.upload_time >= start_date,
            VideoFile.risk_level == 'high'
        ).group_by(cast(VideoFile.upload_time, Date)).all()
        
        # 构建日期系列
        date_series = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                      for i in range(days)][::-1]
        
        # 构建完整数据集
        upload_data = {date: 0 for date in date_series}
        risk_data = {date: 0 for date in date_series}
        
        for date, count in daily_uploads:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in upload_data:
                upload_data[date_str] = count
        
        for date, count in high_risk_daily:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in risk_data:
                risk_data[date_str] = count
        
        return success_response({
            'dates': date_series,
            'uploads': [upload_data[date] for date in date_series],
            'high_risk': [risk_data[date] for date in date_series]
        })
        
    except Exception as e:
        logger.exception(f"获取趋势数据失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")

@analytics_api.route('/api/analytics/content', methods=['GET'])
def get_content_analysis():
    """获取内容分析统计数据"""
    try:
        # 评估项平均分
        assessment_avg = db.session.query(
            func.avg(ContentAnalysis.p1_score).label('p1'),
            func.avg(ContentAnalysis.p2_score).label('p2'),
            func.avg(ContentAnalysis.p3_score).label('p3'),
            func.avg(ContentAnalysis.p4_score).label('p4'),
            func.avg(ContentAnalysis.p5_score).label('p5'),
            func.avg(ContentAnalysis.p6_score).label('p6'),
            func.avg(ContentAnalysis.p7_score).label('p7'),
            func.avg(ContentAnalysis.p8_score).label('p8')
        ).one()
        
        # 风险概率分布
        risk_bins = [
            {'min': 0, 'max': 0.2, 'count': 0},
            {'min': 0.2, 'max': 0.4, 'count': 0},
            {'min': 0.4, 'max': 0.6, 'count': 0},
            {'min': 0.6, 'max': 0.8, 'count': 0},
            {'min': 0.8, 'max': 1.0, 'count': 0}
        ]
        
        # 查询并填充风险概率分布
        for i, bin_range in enumerate(risk_bins):
            count = db.session.query(func.count(ContentAnalysis.id)).filter(
                ContentAnalysis.risk_probability >= bin_range['min'],
                ContentAnalysis.risk_probability < bin_range['max'] 
                if bin_range['max'] < 1.0 else ContentAnalysis.risk_probability <= bin_range['max']
            ).scalar() or 0
            
            risk_bins[i]['count'] = count
        
        return success_response({
            'assessment_avg': {
                'p1': round(assessment_avg.p1 or 0, 2),
                'p2': round(assessment_avg.p2 or 0, 2),
                'p3': round(assessment_avg.p3 or 0, 2),
                'p4': round(assessment_avg.p4 or 0, 2),
                'p5': round(assessment_avg.p5 or 0, 2),
                'p6': round(assessment_avg.p6 or 0, 2),
                'p7': round(assessment_avg.p7 or 0, 2),
                'p8': round(assessment_avg.p8 or 0, 2)
            },
            'risk_probability_distribution': risk_bins
        })
        
    except Exception as e:
        logger.exception(f"获取内容分析统计失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")

@analytics_api.route('/api/analytics/sources', methods=['GET'])
def get_video_sources():
    """获取视频来源统计"""
    try:
        # 平台分布
        platform_stats = db.session.query(
            VideoFile.source_platform, 
            func.count(VideoFile.id).label('count')
        ).filter(VideoFile.source_platform.isnot(None))\
         .group_by(VideoFile.source_platform).all()
        
        # 修改平台分类，添加bilibili
        platforms = {
            'douyin': 0,
            'tiktok': 0,
            'bilibili': 0,  # 添加bilibili
            'upload': 0     # 将local改名为upload
        }
        
        # 计算用户上传数量(原来的local改为upload)
        upload_count = db.session.query(func.count(VideoFile.id))\
            .filter(VideoFile.source_platform.is_(None)).scalar() or 0
        platforms['upload'] = upload_count
        
        # 统计各平台数量
        for platform, count in platform_stats:
            platform_lower = platform.lower()
            if platform_lower in platforms:
                platforms[platform_lower] = count
            # 不再使用other分类，确保所有可能的平台都被正确分类
        
        # 使用数字人概率进行数字人统计
        digital_human_threshold = 0.7
        digital_count = db.session.query(func.count(VideoFile.id))\
            .filter(VideoFile.digital_human_probability >= digital_human_threshold).scalar() or 0
        
        non_digital_count = db.session.query(func.count(VideoFile.id))\
            .filter(VideoFile.digital_human_probability < digital_human_threshold).scalar() or 0
        
        # 未知数量
        unknown_count = db.session.query(func.count(VideoFile.id))\
            .filter(VideoFile.digital_human_probability.is_(None)).scalar() or 0
        
        aigc_data = {
            'digital': digital_count,
            'non_digital': non_digital_count,
            'unknown': unknown_count
        }
        
        return success_response({
            'platform_distribution': platforms,
            'digital_human_distribution': aigc_data
        })
        
    except Exception as e:
        logger.exception(f"获取视频来源统计失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")
@analytics_api.route('/api/analytics/risk-distribution', methods=['GET'])
def get_risk_distribution():
    """获取视频风险分布与数字人关系的统计数据"""
    try:
        # 设置数字人阈值 - 概率大于0.7认为是数字人
        digital_human_threshold = 0.7
        
        # 按风险等级和是否为数字人的分布统计
        risk_distribution = {
            'high': {'digital': 0, 'non_digital': 0},
            'medium': {'digital': 0, 'non_digital': 0},
            'low': {'digital': 0, 'non_digital': 0}
        }
        
        # 查询视频文件的风险等级和数字人概率
        videos = db.session.query(
            VideoFile.risk_level,
            VideoFile.digital_human_probability
        ).filter(VideoFile.risk_level.in_(['high', 'medium', 'low'])).all()
        
        # 统计每个组合的数量
        for risk_level, probability in videos:
            if risk_level not in risk_distribution:
                continue
                
            # 确定是否为数字人 - 基于概率阈值
            is_digital = (probability >= digital_human_threshold) if probability is not None else False
            
            # 更新相应的计数
            if is_digital:
                risk_distribution[risk_level]['digital'] += 1
            else:
                risk_distribution[risk_level]['non_digital'] += 1
        
        # 计算风险等级总数
        totals = {}
        for level in risk_distribution:
            digital = risk_distribution[level]['digital']
            non_digital = risk_distribution[level]['non_digital']
            totals[level] = digital + non_digital
            
        return success_response({
            'risk_distribution': risk_distribution,
            'risk_totals': totals
        })
        
    except Exception as e:
        logger.exception(f"获取风险分布数据失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")    
# 在文件末尾添加以下API端点
@analytics_api.route('/api/analytics/risk-monitor', methods=['GET'])
def get_risk_monitor_data():
    """获取风险监控数据，包括高风险视频、错误事实核查和高风险用户"""
    try:
        # 1. 获取最近的高风险视频（限制10条）
        high_risk_videos = db.session.query(
            VideoFile.id,
            VideoFile.filename,
            VideoFile.source_platform,
            VideoFile.digital_human_probability,
            VideoFile.upload_time,
            VideoFile.summary,
            VideoFile.source_id,  # 增加原始视频ID字段
            DouyinVideo.cover_url  # 增加封面URL字段
        ).outerjoin(
            DouyinVideo, 
            VideoFile.id == DouyinVideo.video_file_id  # 通过外键关联
        ).filter(
            VideoFile.risk_level == 'high'
        ).order_by(VideoFile.upload_time.desc()).limit(10).all()
        
        # 构建高风险视频数据
        recent_risk_videos = []
        for video in high_risk_videos:
            recent_risk_videos.append({
                'id': video.id,
                'filename': video.filename,
                'platform': video.source_platform or '上传',
                'digital_human_probability': video.digital_human_probability or 0,
                'upload_time': video.upload_time.isoformat() if video.upload_time else None,
                'summary': video.summary,
                'cover_url': video.cover_url,  # 添加封面URL
                'source_id': video.source_id  # 添加原始视频ID
            })
        
        # 2. 获取包含不实信息的事实核查结果（限制10条）
        # 这里需要从VideoTranscript表中找出包含假信息的记录
        # 因为fact_check_results是JSON字段，我们需要在应用层面处理
        recent_transcripts = db.session.query(
            VideoTranscript.id,
            VideoTranscript.video_id,
            VideoTranscript.fact_check_results,
            VideoTranscript.fact_check_timestamp,
            VideoFile.filename
        ).join(
            VideoFile, VideoTranscript.video_id == VideoFile.id
        ).filter(
            VideoTranscript.fact_check_status == 'completed',
            VideoTranscript.fact_check_results.isnot(None)
        ).order_by(VideoTranscript.fact_check_timestamp.desc()).limit(30).all()
        
        # 处理事实核查结果，找出包含不实信息的记录
        falsehoods = []
        for transcript in recent_transcripts:
            if not transcript.fact_check_results:
                continue
                
            results = transcript.fact_check_results
            for result in results:
                # 改为使用中文"否"判断不实信息
                if result.get('is_true') == '否':  # 修改这行
                    falsehoods.append({
                        'claim': result.get('claim', ''),
                        'conclusion': result.get('conclusion', ''),
                        'video_id': transcript.video_id,
                        'video_name': transcript.filename,
                        'check_time': transcript.fact_check_timestamp.isoformat() if transcript.fact_check_timestamp else None
                    })
                    if len(falsehoods) >= 10:  # 最多保留10条记录
                        break
        
        # 3. 获取高风险用户排行（根据digital_human_probability和risk_level排序）
        high_risk_users = db.session.query(
            UserAnalysisTask.platform,
            UserAnalysisTask.platform_user_id,
            UserAnalysisTask.nickname,
            UserAnalysisTask.avatar,
            UserAnalysisTask.digital_human_probability,
            UserAnalysisTask.risk_level,
            UserAnalysisTask.completed_at
        ).filter(
            UserAnalysisTask.risk_level == 'high',
            UserAnalysisTask.status == 'completed'
        ).order_by(
            UserAnalysisTask.digital_human_probability.desc(),
            UserAnalysisTask.completed_at.desc()
        ).limit(10).all()
        
        risk_users_data = []
        for user in high_risk_users:
            risk_users_data.append({
                'platform': user.platform,
                'platform_user_id': user.platform_user_id,
                'nickname': user.nickname,
                'avatar': user.avatar,
                'digital_human_probability': user.digital_human_probability,
                'risk_level': user.risk_level,
                'completed_at': user.completed_at.isoformat() if user.completed_at else None
            })
            
        # 4. 获取最近检测出的疑似数字人用户（概率>=0.7）
        digital_human_users = db.session.query(
            UserAnalysisTask.platform,
            UserAnalysisTask.platform_user_id,
            UserAnalysisTask.nickname,
            UserAnalysisTask.avatar,
            UserAnalysisTask.digital_human_probability,
            UserAnalysisTask.completed_at
        ).filter(
            UserAnalysisTask.digital_human_probability >= 0.7,
            UserAnalysisTask.status == 'completed'
        ).order_by(
            UserAnalysisTask.completed_at.desc()
        ).limit(10).all()
        
        dh_users_data = []
        for user in digital_human_users:
            dh_users_data.append({
                'platform': user.platform,
                'platform_user_id': user.platform_user_id,
                'nickname': user.nickname,
                'avatar': user.avatar,
                'digital_human_probability': user.digital_human_probability,
                'completed_at': user.completed_at.isoformat() if user.completed_at else None
            })
            
        return success_response({
            'high_risk_videos': recent_risk_videos,
            'falsehoods': falsehoods,
            'high_risk_users': risk_users_data,
            'digital_human_users': dh_users_data,
        })
        
    except Exception as e:
        logger.exception(f"获取风险监控数据失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")
@analytics_api.route('/api/analytics/system-performance', methods=['GET'])
def get_system_performance():
    """获取系统性能统计数据"""
    try:
        # 1. 任务处理时间统计
        task_performance = db.session.query(
            VideoProcessingTask.task_type,
            func.avg(
                func.unix_timestamp(VideoProcessingTask.completed_at) - 
                func.unix_timestamp(VideoProcessingTask.started_at)
            ).label('avg_duration'),
            func.count(VideoProcessingTask.id).label('total_tasks'),
            func.sum(case((VideoProcessingTask.status == 'completed', 1), else_=0)).label('completed_tasks'),
            func.sum(case((VideoProcessingTask.status == 'failed', 1), else_=0)).label('failed_tasks'),
            func.avg(VideoProcessingTask.attempts).label('avg_attempts')
        ).filter(
            VideoProcessingTask.started_at.isnot(None),
            VideoProcessingTask.completed_at.isnot(None)
        ).group_by(VideoProcessingTask.task_type).all()

        # 格式化任务性能数据
        task_stats = []
        for task in task_performance:
            success_rate = (task.completed_tasks / task.total_tasks * 100) if task.total_tasks > 0 else 0
            task_stats.append({
                'task_type': task.task_type,
                'avg_duration': round(task.avg_duration or 0, 2),
                'total_tasks': task.total_tasks,
                'completed_tasks': task.completed_tasks,
                'failed_tasks': task.failed_tasks,
                'success_rate': round(success_rate, 2),
                'avg_attempts': round(task.avg_attempts or 0, 2)
            })

        # 2. 系统日志统计
        log_stats = db.session.query(
            ProcessingLog.level,
            func.count(ProcessingLog.id).label('count')
        ).group_by(ProcessingLog.level).all()

        log_distribution = {level: count for level, count in log_stats}

        # 3. 最近7天的处理量趋势
        week_ago = datetime.now() - timedelta(days=7)
        daily_processing = db.session.query(
            cast(VideoProcessingTask.completed_at, Date).label('date'),
            func.count(VideoProcessingTask.id).label('completed_count'),
            func.avg(
                func.unix_timestamp(VideoProcessingTask.completed_at) - 
                func.unix_timestamp(VideoProcessingTask.started_at)
            ).label('avg_duration')
        ).filter(
            VideoProcessingTask.completed_at >= week_ago,
            VideoProcessingTask.status == 'completed'
        ).group_by(cast(VideoProcessingTask.completed_at, Date)).all()

        # 构建7天趋势数据
        date_series = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                      for i in range(7)][::-1]
        
        daily_data = {date: {'count': 0, 'avg_duration': 0} for date in date_series}
        for date, count, duration in daily_processing:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in daily_data:
                daily_data[date_str] = {
                    'count': count,
                    'avg_duration': round(duration or 0, 2)
                }

        # 4. 事实核查性能统计
        fact_check_stats = db.session.query(
            func.avg(VideoTranscript.total_search_duration).label('avg_fact_check_duration'),
            func.count(VideoTranscript.id).label('total_fact_checks'),
            func.sum(case((VideoTranscript.fact_check_status == 'completed', 1), else_=0)).label('completed_fact_checks')
        ).filter(VideoTranscript.total_search_duration.isnot(None)).first()

        # 5. 内容分析评估耗时（模拟数据，实际需要记录时间戳）
        assessment_performance = {
            'p1': {'avg_time': 2.5, 'success_rate': 98.5},
            'p2': {'avg_time': 4.2, 'success_rate': 95.8},
            'p3': {'avg_time': 3.1, 'success_rate': 97.2},
            'p4': {'avg_time': 3.8, 'success_rate': 96.1},
            'p5': {'avg_time': 1.9, 'success_rate': 99.1},
            'p6': {'avg_time': 3.5, 'success_rate': 96.8},
            'p7': {'avg_time': 2.8, 'success_rate': 97.5},
            'p8': {'avg_time': 4.1, 'success_rate': 94.9}
        }

        # 6. 系统资源使用概览
        total_processing_tasks = db.session.query(VideoProcessingTask).count()
        total_videos_processed = db.session.query(VideoFile).filter(
            VideoFile.status.in_(['completed', 'processing'])
        ).count()
        
        # 计算平均视频处理时间，使用unix_timestamp
        average_video_processing_time = db.session.query(
            func.avg(
                func.unix_timestamp(VideoProcessingTask.completed_at) - 
                func.unix_timestamp(VideoProcessingTask.started_at)
            )
        ).filter(
            VideoProcessingTask.status == 'completed',
            VideoProcessingTask.started_at.isnot(None),
            VideoProcessingTask.completed_at.isnot(None)
        ).scalar()

        return success_response({
            'task_performance': task_stats,
            'log_distribution': log_distribution,
            'daily_trends': {
                'dates': date_series,
                'processing_counts': [daily_data[date]['count'] for date in date_series],
                'avg_durations': [daily_data[date]['avg_duration'] for date in date_series]
            },
            'fact_check_performance': {
                'avg_duration': round(fact_check_stats.avg_fact_check_duration or 0, 2),
                'total_checks': fact_check_stats.total_fact_checks or 0,
                'success_rate': round(
                    (fact_check_stats.completed_fact_checks / fact_check_stats.total_fact_checks * 100) 
                    if fact_check_stats.total_fact_checks > 0 else 0, 2
                )
            },
            'assessment_performance': assessment_performance,
            'system_overview': {
                'total_processing_tasks': total_processing_tasks,
                'total_videos_processed': total_videos_processed,
                'avg_video_processing_time': round(average_video_processing_time or 0, 2)
            }
        })
        
    except Exception as e:
        logger.exception(f"获取系统性能统计失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")

@analytics_api.route('/api/analytics/error-analysis', methods=['GET'])
def get_error_analysis():
    """获取错误分析统计"""
    try:
        # 1. 错误类型分布
        error_logs = db.session.query(
            ProcessingLog.task_type,
            ProcessingLog.message,
            func.count(ProcessingLog.id).label('count')
        ).filter(
            ProcessingLog.level == 'ERROR'
        ).group_by(
            ProcessingLog.task_type, 
            ProcessingLog.message
        ).order_by(func.count(ProcessingLog.id).desc()).limit(20).all()

        error_distribution = []
        for log in error_logs:
            error_distribution.append({
                'task_type': log.task_type or 'unknown',
                'error_message': log.message[:100] + '...' if len(log.message) > 100 else log.message,
                'count': log.count
            })

        # 2. 失败任务统计
        failed_tasks = db.session.query(
            VideoProcessingTask.task_type,
            func.count(VideoProcessingTask.id).label('failed_count'),
            func.avg(VideoProcessingTask.attempts).label('avg_attempts')
        ).filter(
            VideoProcessingTask.status == 'failed'
        ).group_by(VideoProcessingTask.task_type).all()

        failure_stats = []
        for task in failed_tasks:
            failure_stats.append({
                'task_type': task.task_type,
                'failed_count': task.failed_count,
                'avg_attempts': round(task.avg_attempts or 0, 2)
            })

        # 3. 最近的错误日志
        recent_errors = db.session.query(
            ProcessingLog.task_type,
            ProcessingLog.message,
            ProcessingLog.created_at,
            VideoFile.filename
        ).join(
            VideoFile, ProcessingLog.video_id == VideoFile.id
        ).filter(
            ProcessingLog.level == 'ERROR'
        ).order_by(ProcessingLog.created_at.desc()).limit(10).all()

        recent_error_list = []
        for error in recent_errors:
            recent_error_list.append({
                'task_type': error.task_type or 'unknown',
                'message': error.message,
                'timestamp': error.created_at.isoformat() if error.created_at else None,
                'video_filename': error.filename
            })

        return success_response({
            'error_distribution': error_distribution,
            'failure_stats': failure_stats,
            'recent_errors': recent_error_list
        })

    except Exception as e:
        logger.exception(f"获取错误分析失败: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")