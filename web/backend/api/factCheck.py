import datetime
import logging
from flask import Blueprint, request
from utils.database import VideoFile, VideoTranscript, FactCheckResult, db
from utils.HttpResponse import success_response, error_response
from services.fact_checking.FactCheckPipeline import get_fact_checker

fact_check_api = Blueprint('fact_check', __name__)
logger = logging.getLogger(__name__)

@fact_check_api.route('/api/transcribe/factcheck', methods=['POST'])
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

@fact_check_api.route('/api/videos/<file_id>/factcheck', methods=['POST', 'GET'])
def fact_check_video_transcript(file_id):
    """根据视频ID获取转录内容并进行事实核查"""
    try:
        logging.info(f"开始处理视频 {file_id} 的事实核查请求")
        
        # 查询数据库获取视频转录信息
        transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
        if not transcript:
            return error_response(404, "未找到该视频的转录内容，请先进行转录")
        
        logging.info(f"找到转录记录，ID: {transcript.id}, 当前状态: {transcript.fact_check_status}")
        
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
        
        # 更新数据库中的状态 - 确保事务正确提交
        try:
            logging.info("准备更新事实核查状态为 processing")
            transcript.fact_check_status = "processing"
            transcript.fact_check_context = context
            transcript.fact_check_timestamp = datetime.datetime.utcnow()
            db.session.flush()  # 先flush确保数据写入
            db.session.commit()
            logging.info(f"已更新事实核查状态为 processing")
        except Exception as db_err:
            logging.error(f"更新数据库状态失败: {str(db_err)}")
            db.session.rollback()
            raise db_err
        
        try:
            # 记录开始时间
            start_time = datetime.datetime.now()
            logging.info("开始执行事实核查")
            
            # 获取事实核查器
            fact_checker = get_fact_checker()
            
            # 执行事实核查
            result = fact_checker.run_pipeline(full_text, context)
            logging.info(f"事实核查完成，worth_checking: {result.get('worth_checking')}")
            
            # 计算总用时
            end_time = datetime.datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            logging.info(f"事实核查总用时: {total_duration:.2f} 秒")
            
            # 关键修复：无论worth_checking是什么，都要保存到数据库
            try:
                logging.info("开始保存事实核查结果到数据库")
                
                # 重新查询transcript以确保有最新状态
                transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
                if not transcript:
                    raise Exception("转录记录不存在")
                
                logging.info(f"重新查询转录记录成功，ID: {transcript.id}")
                
                # 先检查当前状态
                logging.info(f"当前fact_check_status: {transcript.fact_check_status}")
                logging.info(f"当前worth_checking: {transcript.worth_checking}")
                
                # 设置基本的事实核查结果 - 无论worth_checking是什么
                transcript.fact_check_status = "completed"
                transcript.worth_checking = result.get('worth_checking', False)
                transcript.worth_checking_reason = result.get('reason', '')
                transcript.claims = result.get('claims', [])
                
                # 设置通用的元数据
                transcript.total_search_duration = total_duration
                transcript.search_metadata = {
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "context": context,
                    "total_duration": total_duration
                }
                
                logging.info(f"设置新的字段值:")
                logging.info(f"  - fact_check_status: completed")
                logging.info(f"  - worth_checking: {result.get('worth_checking', False)}")
                logging.info(f"  - worth_checking_reason: {result.get('reason', '')}")
                logging.info(f"  - claims: {len(result.get('claims', []))} 条")
                
                # 根据worth_checking处理不同情况
                if result.get('worth_checking', False):
                    logging.info("处理值得核查的情况")
                    
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
                    transcript.search_summary = search_summary
                    transcript.search_keywords = " ".join(unique_keywords) if unique_keywords else ""
                    transcript.search_grade = avg_grade
                    
                    # 使用 FactCheckResult 表存储详细结果
                    try:
                        # 清除现有结果
                        deleted_count = db.session.query(FactCheckResult).filter_by(transcript_id=transcript.id).delete()
                        logging.info(f"清除了 {deleted_count} 条旧的核查结果")
                        
                        # 添加新的核查结果
                        for i, check_result in enumerate(result.get('fact_check_results', [])):
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
                            logging.info(f"添加核查结果 {i+1}: {check_result.get('claim', '')[:50]}...")
                            
                        logging.info(f"已将 {len(result.get('fact_check_results', []))} 条核查结果保存到 FactCheckResult 表")
                    except Exception as table_err:
                        logging.warning(f"保存到 FactCheckResult 表失败: {str(table_err)}")
                        logging.exception("FactCheckResult表保存异常详情")
                
                else:
                    logging.info("处理不值得核查的情况")
                    # 不值得核查的情况，设置默认值
                    search_summary = {
                        "total_claims": 0,
                        "true_claims": 0,
                        "false_claims": 0,
                        "uncertain_claims": 0,
                        "error_claims": 0,
                        "keywords": [],
                        "average_search_grade": 0
                    }
                    transcript.search_summary = search_summary
                    transcript.search_keywords = ""
                    transcript.search_grade = 0
                    
                    # 清除可能存在的旧的核查结果
                    deleted_count = db.session.query(FactCheckResult).filter_by(transcript_id=transcript.id).delete()
                    logging.info(f"该内容不值得核查，已清除 {deleted_count} 条旧的核查结果")
                
                logging.info("准备提交数据库事务")
                logging.info(f"即将保存的状态: fact_check_status={transcript.fact_check_status}, worth_checking={transcript.worth_checking}")
                
                # 提交到数据库
                db.session.commit()
                logging.info("数据库提交成功！")
                
                # 验证保存结果
                verify_transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
                logging.info(f"验证保存结果:")
                logging.info(f"  - ID: {verify_transcript.id}")
                logging.info(f"  - fact_check_status: {verify_transcript.fact_check_status}")
                logging.info(f"  - worth_checking: {verify_transcript.worth_checking}")
                logging.info(f"  - worth_checking_reason: {verify_transcript.worth_checking_reason}")
                logging.info(f"  - fact_check_timestamp: {verify_transcript.fact_check_timestamp}")
                
                # 构造响应数据 - 包含search_summary
                response_data = {
                    "video_id": file_id,
                    "filename": video.filename if video else "Unknown",
                    "fact_check_result": {
                        "worth_checking": result.get('worth_checking', False),
                        "reason": result.get('reason', ''),
                        "claims": result.get('claims', []),
                        "fact_check_results": result.get('fact_check_results', []),
                        "search_summary": transcript.search_summary,
                        "metadata": {
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                            "context": context,
                            "total_duration": total_duration
                        }
                    }
                }
                
                logging.info("返回事实核查结果")
                return success_response(response_data)
                
            except Exception as save_err:
                logging.error(f"保存事实核查结果失败: {str(save_err)}")
                logging.exception("保存异常详情")
                db.session.rollback()
                
                # 设置失败状态
                try:
                    transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
                    if transcript:
                        transcript.fact_check_status = "failed"
                        transcript.fact_check_error = str(save_err)
                        db.session.commit()
                        logging.info("已设置失败状态")
                except Exception as final_save_err:
                    logging.error(f"设置失败状态也失败了: {str(final_save_err)}")
                    db.session.rollback()
                
                raise save_err
            
        except Exception as check_err:
            logging.error(f"事实核查执行失败: {str(check_err)}")
            logging.exception("事实核查异常详情")
            
            # 如果事实核查过程中出现错误
            try:
                transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
                if transcript:
                    transcript.fact_check_status = "failed"
                    transcript.fact_check_error = str(check_err)
                    db.session.commit()
                    logging.info("已保存失败状态")
            except Exception as final_err:
                logging.error(f"保存失败状态时出错: {str(final_err)}")
                db.session.rollback()
            raise check_err
    
    except Exception as e:
        logging.exception(f"视频转录事实核查失败: {str(e)}")
        return error_response(500, f"事实核查失败: {str(e)}")

@fact_check_api.route('/api/videos/<file_id>/factcheck/result', methods=['GET'])
def get_fact_check_result(file_id):
    """获取视频的事实核查结果，从新的FactCheckResult表获取详细数据"""
    try:
        logging.info(f"查询视频 {file_id} 的事实核查结果")
        
        # 先检查视频文件是否存在
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        if not video:
            logging.warning(f"未找到视频文件记录: {file_id}")
            return error_response(404, "未找到该视频文件记录")
        
        logging.info(f"找到视频文件: {video.filename}")
        
        # 查询所有与该视频相关的转录记录
        all_transcripts = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).all()
        logging.info(f"找到 {len(all_transcripts)} 条转录记录")
        
        for idx, trans in enumerate(all_transcripts):
            logging.info(f"转录记录 {idx+1}: ID={trans.id}, fact_check_status='{trans.fact_check_status}', worth_checking={trans.worth_checking}")
        
        # 查询数据库获取视频转录信息
        transcript = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).first()
        if not transcript:
            logging.warning(f"未找到视频 {file_id} 的转录记录")
            # 检查是否有转录记录但video_id不匹配
            all_video_transcripts = db.session.query(VideoTranscript).all()
            logging.info(f"数据库中总共有 {len(all_video_transcripts)} 条转录记录")
            for trans in all_video_transcripts[:5]:  # 只显示前5条
                logging.info(f"转录记录: video_id='{trans.video_id}', id={trans.id}")
            return error_response(404, "未找到该视频的转录内容")
        
        # 添加详细的调试信息
        logging.info(f"找到转录记录详情:")
        logging.info(f"  - ID: {transcript.id}")
        logging.info(f"  - video_id: '{transcript.video_id}'")
        logging.info(f"  - fact_check_status: '{transcript.fact_check_status}'")
        logging.info(f"  - worth_checking: {transcript.worth_checking}")
        logging.info(f"  - fact_check_timestamp: {transcript.fact_check_timestamp}")
        logging.info(f"  - worth_checking_reason: {transcript.worth_checking_reason}")
        
        # 状态检查 - 修复这里的逻辑，添加更详细的判断
        current_status = transcript.fact_check_status or "pending"
        logging.info(f"当前状态判断: current_status='{current_status}'")
        
        if current_status in ["pending", None]:
            logging.info(f"事实核查状态为 {current_status}，返回404")
            return error_response(404, "该视频尚未进行事实核查")
            
        if current_status == "processing":
            logging.info("事实核查正在进行中")
            return success_response({
                "video_id": file_id,
                "filename": video.filename if video else "Unknown",
                "status": "processing",
                "message": "事实核查正在进行中"
            })
            
        if current_status == "failed":
            logging.info(f"事实核查失败: {transcript.fact_check_error}")
            return error_response(500, f"事实核查失败: {transcript.fact_check_error}")
        
        # 如果事实核查已完成（包括不值得核查的情况）
        if current_status == "completed":
            logging.info("事实核查已完成，准备返回结果")
            
            # 从FactCheckResult表查询详细结果
            fact_check_results = db.session.query(FactCheckResult).filter_by(transcript_id=transcript.id).all()
            detailed_results = [result.to_dict() for result in fact_check_results]
            
            logging.info(f"从FactCheckResult表查询到 {len(detailed_results)} 条详细结果")
                
            # 构造完整的结果响应
            response_data = {
                "video_id": file_id,
                "filename": video.filename if video else "Unknown",
                "status": current_status,
                "timestamp": transcript.fact_check_timestamp.isoformat() if transcript.fact_check_timestamp else None,
                "worth_checking": transcript.worth_checking if transcript.worth_checking is not None else False,
                "reason": transcript.worth_checking_reason or "系统判断该内容不需要进行事实核查",
                "context": transcript.fact_check_context,
                "claims": transcript.claims or [],
                "fact_check_results": detailed_results,  # 使用新表中的详细结果
                
                # 使用VideoTranscript表中保存的汇总信息
                "search_summary": transcript.search_summary or {
                    "total_claims": 0,
                    "true_claims": 0,
                    "false_claims": 0,
                    "uncertain_claims": 0,
                    "error_claims": 0,
                    "keywords": [],
                    "average_search_grade": 0
                },
                "total_search_duration": transcript.total_search_duration,
                "search_metadata": transcript.search_metadata or {}
            }
            
            logging.info(f"返回事实核查结果: worth_checking={response_data['worth_checking']}")
            return success_response(response_data)
        
        # 如果状态不明确，返回错误
        logging.error(f"未知的事实核查状态: {current_status}")
        return error_response(500, f"未知的事实核查状态: {current_status}")
        
    except Exception as e:
        logging.exception(f"获取事实核查结果失败: {str(e)}")
        return error_response(500, f"获取事实核查结果失败: {str(e)}")

# 添加一个新的调试接口来检查数据库状态
@fact_check_api.route('/api/videos/<file_id>/factcheck/debug', methods=['GET'])
def debug_fact_check_data(file_id):
    """调试接口：检查视频的转录和事实核查数据"""
    try:
        logging.info(f"=== 调试视频 {file_id} 的数据 ===")
        
        # 检查视频文件  
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        video_info = {
            "exists": video is not None,
            "filename": video.filename if video else None,
            "id": video.id if video else None
        }
        logging.info(f"视频文件: {video_info}")
        
        # 检查转录记录
        transcripts = db.session.query(VideoTranscript).filter(VideoTranscript.video_id == file_id).all()
        transcript_info = []
        for trans in transcripts:
            transcript_info.append({
                "id": trans.id,
                "video_id": trans.video_id,
                "fact_check_status": trans.fact_check_status,
                "worth_checking": trans.worth_checking,
                "fact_check_timestamp": trans.fact_check_timestamp.isoformat() if trans.fact_check_timestamp else None,
                "has_transcript": bool(trans.transcript and trans.transcript.strip()),
                "transcript_length": len(trans.transcript) if trans.transcript else 0
            })
        
        logging.info(f"转录记录: {transcript_info}")
        
        # 检查事实核查结果
        fact_check_info = []
        for trans in transcripts:
            results = db.session.query(FactCheckResult).filter_by(transcript_id=trans.id).all()
            fact_check_info.append({
                "transcript_id": trans.id,
                "result_count": len(results),
                "results": [{"id": r.id, "claim": r.claim[:50] + "..." if len(r.claim) > 50 else r.claim, "is_true": r.is_true} for r in results]
            })
        
        logging.info(f"事实核查结果: {fact_check_info}")
        
        debug_data = {
            "video_id": file_id,
            "video": video_info,
            "transcripts": transcript_info,
            "fact_check_results": fact_check_info,
            "total_transcripts_in_db": db.session.query(VideoTranscript).count(),
            "total_videos_in_db": db.session.query(VideoFile).count()
        }
        
        return success_response(debug_data)
        
    except Exception as e:
        logging.exception(f"调试接口失败: {str(e)}")
        return error_response(500, f"调试失败: {str(e)}")