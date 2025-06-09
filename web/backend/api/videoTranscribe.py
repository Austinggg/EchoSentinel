import datetime
from flask import Blueprint, request, jsonify
from api.videoUpload import get_video_file_path
from werkzeug.utils import secure_filename
from utils.database import db
from utils.HttpResponse import success_response, error_response
from services.fact_checking.FactCheckPipeline import get_fact_checker
import os
import tempfile
import logging

from utils import HttpResponse

transcribe_api = Blueprint('transcribe', __name__)

# 初始化转录器
from services.content_analysis.video_transcribe import VideoTranscriber  # 调整导入路径
transcriber = VideoTranscriber()

@transcribe_api.route('/api/transcribe/file', methods=['POST'])
def transcribe_file():
    """
    处理单个视频文件上传并返回转录结果
    """
    try:
        # 检查文件上传
        if 'file' not in request.files:
            return error_response(400, "未上传文件")
        
        file = request.files['file']
        if file.filename == '':
            return error_response(400, "无效文件名")
        
        # 验证文件类型
        allowed_extensions = {'mp4', 'avi', 'mkv', 'mov', 'flv'}
        if '.' not in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return error_response(400, "不支持的文件类型")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        temp_video = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_video)
        
        # 处理参数
        keep_audio = request.form.get('keep_audio', 'false').lower() == 'true'
        
        # 执行转录
        result = transcriber.transcribe_video(temp_video)
        
        # 清理临时文件
        if result and not keep_audio:
            audio_path = os.path.splitext(temp_video)[0] + "_audio.wav"
            if os.path.exists(audio_path):
                os.remove(audio_path)
        os.remove(temp_video)
        
        if not result:
            return error_response(500, "视频转录失败")
            
        # 修复这一部分：正确处理全文
        response_data = {
            "filename": file.filename,
            "chunks": result.get("chunks", []),
            "full_text": result.get("text", ""),  # 修改这里，使用text字段
            "audio_path": audio_path if keep_audio else None
        }
        
        return success_response(response_data)
    
    except Exception as e:
        logging.exception("文件处理异常")
        return error_response(500, f"处理失败: {str(e)}")

"""通过ID转录视频"""
@transcribe_api.route("/api/videos/<file_id>/transcribe", methods=["POST"])
def transcribe_video_by_id(file_id):
    try:
        from utils.database import VideoFile, VideoTranscript
        
        # 查询数据库获取视频信息
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        if not video:
            return error_response(404, "视频不存在")
        
        # 获取视频文件路径
        video_path = get_video_file_path(file_id, video.extension)
        if not video_path or not video_path.exists():
            return error_response(404, "视频文件不存在")
        
        # 检查文件大小
        file_size = video_path.stat().st_size
        print(f"视频文件大小: {file_size}")
        if file_size == 0:
            return error_response(400, "视频文件为空")
        
        # 执行转录
        print(f"开始转录视频: {str(video_path)}")
        result = transcriber.transcribe_video(str(video_path))
        
        if not result:
            return error_response(500, "视频转录失败，请检查服务器日志")
            
        # 如果视频没有音频轨道
        if result.get("text", "") == "" and result.get("message") == "视频不包含音频轨道":
            response_data = {
                "video_id": file_id,
                "filename": video.filename,
                "chunks": [],
                "full_text": "视频不包含音频轨道",
                "duration": 0,
                "message": "视频不包含音频轨道，无法转录"
            }
            return success_response(response_data)  # 返回成功但内容为空
            
        
        # 构建响应数据
        response_data = {
            "video_id": file_id,
            "filename": video.filename,
            "chunks": result.get("chunks", []),
            "full_text": result.get("text", ""),
            "duration": result.get("duration", 0)
        }
        
        # 更新或创建转录记录
        save_transcript_to_db(file_id, result, video)
        
        return success_response(response_data)
        
    except Exception as e:
        print(f"视频转录失败: {str(e)}")
        return error_response(500, f"视频转录失败: {str(e)}")
@transcribe_api.route('/api/transcribe/factcheck', methods=['POST'])
def fact_check_text():
    """对提供的文本进行事实核查"""
    try:
        # 检查请求数据
        if not request.is_json:
            return error_response(400, "请求必须是JSON格式，请设置Content-Type: application/json")
        
        data = request.json
        text = data.get('text')
        context = data.get('context')  # 可选参数
        
        if not text:
            return error_response(400, "文本内容不能为空")
        
        # 获取事实核查器
        fact_checker = get_fact_checker()
        
        # 执行事实核查
        result = fact_checker.run_pipeline(text, context)
        
        # 日志记录
        logging.info(f"事实核查结果: {'值得核查' if result.get('worth_checking') else '不值得核查'}")
        if result.get('worth_checking'):
            logging.info(f"找到 {len(result.get('claims', []))} 条需要核查的断言")
        
        return success_response(result)
    
    except Exception as e:
        logging.exception("事实核查处理异常")
        return error_response(500, f"处理失败: {str(e)}")

@transcribe_api.route('/api/videos/<file_id>/factcheck', methods=['POST', 'GET'])
def fact_check_video_transcript(file_id):
    """根据视频ID获取转录内容并进行事实核查"""
    try:
        from utils.database import VideoFile, VideoTranscript, FactCheckResult, db
        import datetime
        import logging
        
        # 查询数据库获取视频转录信息
        transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
        if not transcript:
            return error_response(404, "未找到该视频的转录内容，请先进行转录")
        
        # 从数据库获取视频信息(用于添加上下文)
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        
        # 准备上下文信息
        video_context = f"视频名称: {video.filename}" if video else ""
        tags_context = f"视频标签: {video.tags}" if video and video.tags else ""
        context = f"{video_context} {tags_context}".strip()
        
        # 检查是否有 JSON 数据，如果有则提取 context
        if request.is_json and request.get_data(as_text=True):
            try:
                data = request.get_json()
                if data and 'context' in data:
                    context = data['context']
            except Exception as json_err:
                logging.warning(f"JSON 解析错误: {str(json_err)}")
        
        # 获取转录文本
        full_text = transcript.transcript
        if not full_text or full_text.strip() == "":
            return error_response(400, "转录文本为空，无法进行事实核查")
        
        # 更新数据库中的状态
        transcript.fact_check_status = "processing"
        transcript.fact_check_context = context
        transcript.fact_check_timestamp = datetime.datetime.utcnow()
        db.session.commit()
        
        try:
            # 记录开始时间
            start_time = datetime.datetime.now()
            
            # 获取事实核查器
            fact_checker = get_fact_checker()
            
            # 执行事实核查
            result = fact_checker.run_pipeline(full_text, context)
            
            # 计算总用时
            end_time = datetime.datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # 保存事实核查结果到数据库
            transcript.fact_check_status = "completed"
            transcript.worth_checking = result.get('worth_checking', False)
            transcript.worth_checking_reason = result.get('reason', '')
            transcript.claims = result.get('claims', [])
            
            # 提取关键词和搜索分数
            all_keywords = []
            total_grade = 0
            
            # 处理每个核查结果，提取关键词和分数
            for check_result in result.get('fact_check_results', []):
                if 'search_details' in check_result and check_result['search_details']:
                    # 提取关键词
                    if 'keywords' in check_result['search_details']:
                        keywords = check_result['search_details']['keywords']
                        if keywords:
                            if isinstance(keywords, str):
                                all_keywords.extend(keywords.split())
                            elif isinstance(keywords, list):
                                all_keywords.extend(keywords)
                    
                    # 提取分数
                    if 'grade' in check_result['search_details']:
                        total_grade += check_result['search_details']['grade']
            
            # 计算平均分数
            avg_grade = total_grade / len(result.get('fact_check_results', [])) if result.get('fact_check_results', []) else 0
            unique_keywords = list(set(all_keywords))
            
            # 创建搜索摘要
            search_summary = {
                "total_claims": len(result.get('fact_check_results', [])),
                "true_claims": sum(1 for r in result.get('fact_check_results', []) if r.get("is_true") == "是"),
                "false_claims": sum(1 for r in result.get('fact_check_results', []) if r.get("is_true") == "否"),
                "uncertain_claims": sum(1 for r in result.get('fact_check_results', []) if r.get("is_true") not in ["是", "否", "错误"]),
                "error_claims": sum(1 for r in result.get('fact_check_results', []) if r.get("is_true") == "错误"),
                "keywords": unique_keywords,
                "average_search_grade": avg_grade
            }
            
            # 保存到视频转录表中
            transcript.fact_check_results = result.get('fact_check_results', [])  # 兼容性保留
            transcript.search_keywords = " ".join(unique_keywords) if unique_keywords else ""
            transcript.search_grade = avg_grade
            transcript.search_summary = search_summary
            transcript.total_search_duration = total_duration
            transcript.search_metadata = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "context": context,
                "total_duration": total_duration
            }
            
            # 使用 FactCheckResult 表存储详细结果
            try:
                # 清除现有结果
                db.session.query(FactCheckResult).filter_by(transcript_id=transcript.id).delete()
                
                # 添加新的核查结果
                for check_result in result.get('fact_check_results', []):
                    fact_check = FactCheckResult(
                        transcript_id=transcript.id,
                        claim=check_result.get('claim', ''),
                        is_true=check_result.get('is_true', '未确定'),
                        conclusion=check_result.get('conclusion', ''),
                        search_duration=check_result.get('search_duration', 0),
                        search_query=check_result.get('search_query', ''),
                        search_details=check_result.get('search_details', {})
                    )
                    db.session.add(fact_check)
                    
                logging.info(f"已将 {len(result.get('fact_check_results', []))} 条核查结果保存到 FactCheckResult 表")
            except Exception as table_err:
                logging.warning(f"保存到 FactCheckResult 表失败，将结果保存到主表: {str(table_err)}")
            
            db.session.commit()
            
            # 构造响应数据
            response_data = {
                "video_id": file_id,
                "filename": video.filename if video else "Unknown",
                "fact_check_result": {
                    "worth_checking": result.get('worth_checking', False),
                    "reason": result.get('reason', ''),
                    "claims": result.get('claims', []),
                    "fact_check_results": result.get('fact_check_results', []),
                    "search_keywords": " ".join(unique_keywords) if unique_keywords else "",
                    "search_grade": avg_grade,
                    "search_summary": search_summary,
                    "metadata": {
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "context": context,
                        "total_duration": total_duration
                    }
                }
            }
            
            return success_response(response_data)
            
        except Exception as check_err:
            # 如果事实核查过程中出现错误
            transcript.fact_check_status = "failed"
            transcript.fact_check_error = str(check_err)
            db.session.commit()
            logging.error(f"事实核查失败: {str(check_err)}")
            raise check_err
    
    except Exception as e:
        logging.exception(f"视频转录事实核查失败: {str(e)}")
        return error_response(500, f"事实核查失败: {str(e)}")

@transcribe_api.route('/api/videos/<file_id>/factcheck/result', methods=['GET'])
def get_fact_check_result(file_id):
    """获取视频的事实核查结果，从新的FactCheckResult表获取详细数据"""
    try:
        from utils.database import VideoTranscript, VideoFile, FactCheckResult, db
        
        # 查询数据库获取视频转录信息
        transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
        if not transcript:
            return error_response(404, "未找到该视频的转录内容")
        
        # 获取关联的视频信息
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        
        # 状态检查
        if transcript.fact_check_status == "pending":
            return error_response(404, "该视频尚未进行事实核查")
            
        if transcript.fact_check_status == "processing":
            return success_response({
                "video_id": file_id,
                "filename": video.filename if video else "Unknown",
                "status": "processing",
                "message": "事实核查正在进行中"
            })
            
        if transcript.fact_check_status == "failed":
            return error_response(500, f"事实核查失败: {transcript.fact_check_error}")
        
        # 从FactCheckResult表查询详细结果
        fact_check_results = db.session.query(FactCheckResult).filter_by(transcript_id=transcript.id).all()
        detailed_results = [result.to_dict() for result in fact_check_results]
            
        # 构造完整的结果响应
        response_data = {
            "video_id": file_id,
            "filename": video.filename if video else "Unknown",
            "status": transcript.fact_check_status,
            "timestamp": transcript.fact_check_timestamp.isoformat() if transcript.fact_check_timestamp else None,
            "worth_checking": transcript.worth_checking,
            "reason": transcript.worth_checking_reason,
            "context": transcript.fact_check_context,
            "claims": transcript.claims,
            "fact_check_results": detailed_results,  # 使用新表中的详细结果
            
            # 使用VideoTranscript表中保存的汇总信息
            "search_summary": transcript.search_summary,
            "total_search_duration": transcript.total_search_duration,
            "search_metadata": transcript.search_metadata
        }
        
        return success_response(response_data)
        
    except Exception as e:
        logging.exception(f"获取事实核查结果失败: {str(e)}")
        return error_response(500, f"获取事实核查结果失败: {str(e)}")
# 添加一个函数用于保存转录结果到数据库
def save_transcript_to_db(video_id, result, video):
    """保存转录结果到数据库"""
    from utils.database import VideoTranscript, db
    
    if "text" in result:
        # 查找是否存在现有记录
        transcript = db.session.query(VideoTranscript).filter_by(video_id=video_id).first()
        
        if transcript:
            # 更新现有记录
            transcript.transcript = result["text"]
            transcript.chunks = result.get("chunks", [])
        else:
            # 创建新记录
            transcript = VideoTranscript(
                video_id=video_id,
                transcript=result["text"],
                chunks=result.get("chunks", [])
            )
            db.session.add(transcript)
        
        # 同时更新视频表中的字段以保持兼容性
        video.transcript = result["text"]
        db.session.commit()
@transcribe_api.route('/api/factcheck/claims', methods=['POST'])
def fact_check_claims_api():
    """
    API端点: 单独处理事实核查断言
    
    请求格式:
    {
        "claims": ["断言1", "断言2", ...],
        "context": "可选的上下文信息",
        "original_text": "可选的原始文本"
    }
    """
    try:
        # 检查请求数据
        if not request.is_json:
            return error_response(400, "请求必须是JSON格式，请设置Content-Type: application/json")
        
        data = request.json
        claims = data.get('claims')
        context = data.get('context')  # 可选参数
        original_text = data.get('original_text')  # 可选参数
        
        if not claims or not isinstance(claims, list):
            return error_response(400, "必须提供断言列表，且格式为数组")
        
        if len(claims) == 0:
            return error_response(400, "断言列表不能为空")
        
        # 获取事实核查器
        fact_checker = get_fact_checker()
        
        # 记录开始时间
        start_time = datetime.datetime.now()
        
        # 执行断言核查
        results = fact_checker.fact_check_claims(claims, context, original_text)
        
        # 计算总用时
        end_time = datetime.datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # 汇总搜索信息
        search_summary = {
            "total_claims": len(claims),
            "true_claims": sum(1 for r in results if r.get("is_true") == "是"),
            "false_claims": sum(1 for r in results if r.get("is_true") == "否"),
            "uncertain_claims": sum(1 for r in results if r.get("is_true") not in ["是", "否", "错误"]),
            "error_claims": sum(1 for r in results if r.get("is_true") == "错误"),
            "keywords": list(set(sum([r.get("search_details", {}).get("keywords", "").split() for r in results], []))),
            "average_search_grade": sum(r.get("search_details", {}).get("grade", 0) for r in results) / len(results) if results else 0
        }
        
        # 日志记录
        logging.info(f"完成 {len(claims)} 条断言的事实核查，用时 {total_duration:.2f} 秒")
        
        # 构造响应数据
        response_data = {
            "total": len(claims),
            "results": results,
            "search_summary": search_summary,
            "metadata": {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "context": context if context else None,
                "total_duration": total_duration
            }
        }
        
        return success_response(response_data)
    
    except Exception as e:
        logging.exception("事实核查断言处理异常")
        return error_response(500, f"处理失败: {str(e)}")