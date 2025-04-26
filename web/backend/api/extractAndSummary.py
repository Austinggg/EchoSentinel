# filepath: [extractAndSummary.py](http://_vscodecontentref_/1)
import datetime
from flask import Blueprint, request, jsonify
from utils.database import db
from utils.HttpResponse import success_response, error_response
import logging

# 创建蓝图
extract_api = Blueprint("extract", __name__)

# 引入信息提取器
from services.content_analysis.information_extractor import InformationExtractor

# 初始化信息提取器
information_extractor = InformationExtractor()


@extract_api.route("/api/extract/info", methods=["POST"])
def extract_information():
    """
    从文本中提取关键信息和摘要
    body：json text
    """
    try:
        # 获取请求数据
        data = request.json
        if not data or "text" not in data:
            return error_response(400, "未提供文本内容")

        text = data["text"]
        if not text.strip():
            return error_response(400, "文本内容不能为空")

        # 提取信息
        result = information_extractor.extract_information(text)

        if not result:
            return error_response(500, "信息提取失败")

        # 返回结果
        return success_response(result)

    except Exception as e:
        logging.exception("信息提取异常")
        return error_response(500, f"处理失败: {str(e)}")


@extract_api.route("/api/extract/video/<file_id>", methods=["POST"])
def extract_from_video(file_id):
    """
    从视频转录文本中提取关键信息（不包含摘要生成）
    """
    try:
        from utils.database import VideoFile, VideoTranscript, ContentAnalysis
        import datetime

        # 检查视频是否存在
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        if not video:
            return error_response(404, "视频不存在")

        # 检查是否已有转录文本
        transcript = (
            db.session.query(VideoTranscript)
            .filter(VideoTranscript.video_id == file_id)
            .first()
        )
        if not transcript or not transcript.transcript:
            return error_response(400, "视频尚未转录或没有可用的转录文本")

        # 提取信息
        result = information_extractor.extract_information(transcript.transcript)

        if not result:
            return error_response(500, "信息提取失败")

        # 保存到内容分析表，不包含摘要
        existing_analysis = (
            db.session.query(ContentAnalysis)
            .filter(ContentAnalysis.video_id == file_id)
            .first()
        )

        if existing_analysis:
            # 更新现有记录
            existing_analysis.intent = result.get("intent", [])
            existing_analysis.statements = result.get("statements", [])
            existing_analysis.updated_at = datetime.datetime.utcnow()
        else:
            # 创建新记录
            analysis = ContentAnalysis(
                video_id=file_id,
                intent=result.get("intent", []),
                statements=result.get("statements", []),
            )
            db.session.add(analysis)

        # 更新视频状态，但不更新摘要
        video.status = "analyzed"
        db.session.commit()

        # 返回结果
        return success_response(
            {"video_id": file_id, "filename": video.filename, "analysis": result}
        )

    except Exception as e:
        logging.exception("视频信息提取异常")
        return error_response(500, f"处理失败: {str(e)}")


@extract_api.route("/api/analysis/video/<file_id>", methods=["GET"])
def get_video_analysis(file_id):
    """
    获取视频的内容分析结果
    """
    try:
        from utils.database import VideoFile, ContentAnalysis

        # 查询分析结果
        analysis = (
            db.session.query(ContentAnalysis)
            .filter(ContentAnalysis.video_id == file_id)
            .first()
        )
        if not analysis:
            return error_response(404, "未找到该视频的分析结果")

        # 获取视频信息
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        if not video:
            return error_response(404, "视频不存在")

        # 构建响应
        response_data = {
            "video_id": file_id,
            "filename": video.filename,
            "analysis": {
                "intent": analysis.intent,
                "statements": analysis.statements,
                "summary": analysis.summary,
            },
        }

        return success_response(response_data)

    except Exception as e:
        logging.exception("获取分析结果异常")
        return error_response(500, f"获取失败: {str(e)}")


@extract_api.route("/api/summary/video/<file_id>", methods=["POST"])
def generate_video_summary(file_id):
    """
    为视频内容生成摘要并更新到数据库
    """
    try:
        from utils.database import VideoFile, ContentAnalysis, VideoTranscript
        from services.content_analysis.summary_generator import SummaryGenerator

        # 初始化摘要生成器
        summary_generator = SummaryGenerator()

        # 检查视频是否存在
        video = db.session.query(VideoFile).filter(VideoFile.id == file_id).first()
        if not video:
            return error_response(404, "视频不存在")

        # 检查是否有内容分析记录
        analysis = (
            db.session.query(ContentAnalysis)
            .filter(ContentAnalysis.video_id == file_id)
            .first()
        )
        if not analysis:
            return error_response(404, "未找到视频的内容分析记录，请先进行内容提取")

        # 获取转录文本
        transcript = (
            db.session.query(VideoTranscript)
            .filter(VideoTranscript.video_id == file_id)
            .first()
        )
        if not transcript or not transcript.transcript:
            return error_response(400, "视频尚未转录或没有可用的转录文本")

        # 获取请求中的自定义参数（修改这里，增加容错处理）
        try:
            data = request.get_json(silent=True) or {}
        except:
            data = {}

        # 尝试从URL参数或表单数据获取max_length
        max_length = request.args.get("max_length") or request.form.get("max_length")
        if max_length and str(max_length).isdigit():
            max_length = int(max_length)
        else:
            max_length = data.get("max_length", 500)  # 默认值

        # 从分析表中提取意图和陈述
        extracted_info = {"intent": analysis.intent, "statements": analysis.statements}

        # 生成摘要
        summary = summary_generator.generate_summary(
            transcript.transcript, extracted_info, max_length
        )

        # 更新内容分析表
        analysis.summary = summary
        analysis.updated_at = datetime.datetime.utcnow()

        # 更新视频记录中的简要摘要
        video.summary = summary[:200] + "..." if len(summary) > 200 else summary
        db.session.commit()

        # 返回结果
        return success_response(
            {"video_id": file_id, "filename": video.filename, "summary": summary}
        )

    except Exception as e:
        logging.exception("摘要生成异常")
        return error_response(500, f"处理失败: {str(e)}")
