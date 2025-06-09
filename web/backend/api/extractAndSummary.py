# filepath: [extractAndSummary.py](http://_vscodecontentref_/1)
import datetime
import re
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
        import re

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

        # 参数处理
        try:
            data = request.get_json(silent=True) or {}
        except:
            data = {}

        max_length = request.args.get("max_length") or request.form.get("max_length")
        if max_length and str(max_length).isdigit():
            max_length = int(max_length)
        else:
            max_length = data.get("max_length", 500)  # 默认值

        # 从分析表中提取信息
        extracted_info = {"intent": analysis.intent, "statements": analysis.statements}

        # 生成摘要
        summary = summary_generator.generate_summary(
            transcript.transcript, extracted_info, max_length
        )

        # 更新内容分析表 - 保留完整markdown格式
        analysis.summary = summary
        analysis.updated_at = datetime.datetime.utcnow()

        # 处理用于表格显示的纯文本摘要
        plain_summary = clean_markdown_for_table(summary)
        
        # 更新视频记录中的简要摘要 - 使用纯文本格式
        video.summary = plain_summary[:200] + "..." if len(plain_summary) > 200 else plain_summary
        db.session.commit()

        # 返回结果
        return success_response(
            {
                "video_id": file_id, 
                "filename": video.filename, 
                "summary": summary,
                "plain_summary": plain_summary[:200] + "..." if len(plain_summary) > 200 else plain_summary
            }
        )

    except Exception as e:
        logging.exception("摘要生成异常")
        return error_response(500, f"处理失败: {str(e)}")


def clean_markdown_for_table(text):
    """
    清理Markdown标记和标题，提取纯文本内容用于表格展示
    
    Args:
        text: 原始Markdown格式文本
        
    Returns:
        清理后的纯文本
    """
    if not text:
        return ""
    
    # 移除Markdown标题标记 (## 和 ###)
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    
    # 移除常见标题词 ("内容摘要", "主题", "意图", "核心内容" 等)
    title_patterns = [
        r'内容摘要', r'主题', r'意图', r'核心内容', 
        r'📝.*?[:：]?', r'💡.*?[:：]?', r'✅.*?[:：]?'
    ]
    for pattern in title_patterns:
        text = re.sub(pattern, '', text)
    
    # 移除emoji和其他特殊符号
    text = re.sub(r'[📌📝💡✅🌟]', '', text)
    
    # 移除多余空行
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # 移除行首空白
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    # 连接所有段落成为一段纯文本
    text = re.sub(r'\n+', ' ', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text