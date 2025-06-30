import datetime
import json
from typing import Dict, Optional
from urllib.parse import quote_plus  # 用于密码编码

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from werkzeug.security import check_password_hash, generate_password_hash

from utils.extensions import db


class User(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(256))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"User(id={self.id!r}, username={self.username!r}), password_hash={self.password_hash!r}"


class UserProfile(db.Model):
    __tablename__ = "user_profiles"  # 表名

    id: Mapped[Optional[int]] = mapped_column(primary_key=True)
    sec_uid: Mapped[str] = mapped_column(String(255), index=True)
    hash_sec_uid: Mapped[str] = mapped_column(String(255))
    nickname: Mapped[str] = mapped_column(String(255))
    signature: Mapped[Optional[str]] = mapped_column(String(500))  # 添加用户签名字段
    gender: Mapped[Optional[str]] = mapped_column(String(50))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    province: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    aweme_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    follower_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    following_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    total_favorited: Mapped[Optional[int]] = mapped_column(BigInteger)
    favoriting_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    user_age: Mapped[Optional[int]] = mapped_column(BigInteger)
    ip_location: Mapped[Optional[str]] = mapped_column(String(100))
    show_favorite_list: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_gov_media_vip: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_mix_user: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_star: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_series_user: Mapped[Optional[bool]] = mapped_column(Boolean)
    covers: Mapped[Dict] = mapped_column(JSON, default={})
    avatar_medium: Mapped[Optional[str]] = mapped_column(String(255))
    # 新增加的字段
    district: Mapped[Optional[str]] = mapped_column(String(100))  # 区/县信息
    custom_verify: Mapped[Optional[str]] = mapped_column(String(255))  # 自定义认证信息
    enterprise_verify_reason: Mapped[Optional[str]] = mapped_column(String(255))  # 企业认证原因
    verification_type: Mapped[Optional[int]] = mapped_column(Integer, default=0)  # 认证类型
    # 添加与VideoFile的关系
    videos = relationship("VideoFile", back_populates="user")

    def to_dict(self):
        covers_data = json.loads(self.covers) if self.covers else {}
        return {
            "id": self.id,
            "sec_uid": self.sec_uid,
            "hash_sec_uid": self.hash_sec_uid,
            "nickname": self.nickname,
            "signature": self.signature,
            "gender": self.gender,
            "city": self.city or "未知",
            "province": self.province or "未知",
            "country": self.country or "未知",
            "district": self.district or "未知",  # 添加district
            "aweme_count": self.aweme_count,
            "follower_count": self.follower_count,
            "following_count": self.following_count,
            "total_favorited": self.total_favorited,
            "favoriting_count": self.favoriting_count,
            "user_age": self.user_age,
            "ip_location": self.ip_location,
            "show_favorite_list": self.show_favorite_list,
            "is_gov_media_vip": self.is_gov_media_vip,
            "is_mix_user": self.is_mix_user,
            "is_star": self.is_star,
            "is_series_user": self.is_series_user,
            "custom_verify": self.custom_verify,  # 添加认证信息
            "enterprise_verify_reason": self.enterprise_verify_reason,
            "verification_type": self.verification_type,
            "covers": [
                f"http://localhost:8000/api/userAnalyse/getCover/{x}.jpg"
                for x in covers_data.values()
            ],
            "avatar_medium": self.avatar_medium
        }
# 视频文件表(分析结果表)
class VideoFile(db.Model):
    __tablename__ = "video_files"

    id = mapped_column(String(36), primary_key=True)
    filename = mapped_column(String(255), nullable=False)
    extension = mapped_column(String(10))  # 只存储文件扩展名
    size = mapped_column(Integer)
    mime_type = mapped_column(String(100))
    upload_time = mapped_column(DateTime, default=datetime.datetime.utcnow)
    user_id = mapped_column(Integer, ForeignKey("user_profiles.id"), nullable=True)

    # 分析结果相关字段
    summary = mapped_column(String(500), default="处理中")
    risk_level = mapped_column(String(20), default="processing")
    status = mapped_column(String(50), default="processing")

    # 新增爬虫数据字段
    publish_time = mapped_column(DateTime, nullable=True)  # 视频发布时间
    tags = mapped_column(String(500), nullable=True)  # 视频标签，以逗号分隔

    # 关联的用户
    user = relationship("UserProfile", back_populates="videos")
    # 添加视频来源相关字段
    source_url = mapped_column(String(500), nullable=True)  # 源视频URL
    source_platform = mapped_column(String(50), nullable=True)  # 源平台（douyin/tiktok）
    source_id = mapped_column(String(100), nullable=True)  # 平台上的原始ID

    # 添加数字人检测字段
    # aigc_use=mapped_column(String(50),nullable=True)
    # aigc_face = mapped_column(String(50), nullable=True)
    # aigc_body = mapped_column(String(50), nullable=True)
    # aigc_whole = mapped_column(String(50), nullable=True)
    digital_human_probability = mapped_column(db.Float, default=0.0, nullable=False)
    def to_dict(self):
        tags_list = self.tags.split(",") if self.tags else []
        return {
            "id": self.id,
            "filename": self.filename,
            "size": self.size,
            "mimeType": self.mime_type,
            "uploadTime": self.upload_time.isoformat(),
            "publishTime": self.publish_time.isoformat() if self.publish_time else None,
            "summary": self.summary,
            "riskLevel": self.risk_level,
            "status": self.status,
            "tags": tags_list,
            "url": f"/api/videos/{self.id}",
        }


# 修改 VideoTranscript 类，添加事实核查相关字段

class VideoTranscript(db.Model):
    __tablename__ = "video_transcripts"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    transcript = db.Column(db.Text, nullable=True)  # 完整转录文本
    chunks = db.Column(db.JSON, nullable=True)  # 带时间戳的分段文本
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 事实核查相关字段 - 修复默认值设置
    fact_check_status = db.Column(db.String(20), default="pending", nullable=False)  # 确保不为空
    fact_check_timestamp = db.Column(db.DateTime, nullable=True)  # 事实核查执行时间
    worth_checking = db.Column(db.Boolean, nullable=True)  # 是否值得核查
    worth_checking_reason = db.Column(db.Text, nullable=True)  # 判断原因
    claims = db.Column(db.JSON, nullable=True)  # 提取的断言列表 - 修复默认值
    fact_check_results = db.Column(db.JSON, nullable=True)  # 核查结果列表 - 修复默认值
    fact_check_context = db.Column(db.Text, nullable=True)  # 核查时的上下文信息
    fact_check_error = db.Column(db.Text, nullable=True)  # 如果有错误，记录错误信息
    
    # 添加新字段用于存储搜索结果 - 修复默认值
    search_summary = db.Column(db.JSON, nullable=True)  # 存储搜索摘要
    total_search_duration = db.Column(db.Float, nullable=True)  # 事实核查总耗时
    search_metadata = db.Column(db.JSON, nullable=True)  # 存储元数据
    search_keywords = db.Column(db.Text, nullable=True)  # 添加搜索关键词字段
    search_grade = db.Column(db.Float, nullable=True)  # 添加搜索评分字段
    
    # 关联的视频
    video = db.relationship(
        "VideoFile", backref=db.backref("transcript_data", uselist=False)
    )

    def to_dict(self):
        """转换为字典，方便API返回，增加新字段"""
        return {
            "id": self.id,
            "video_id": self.video_id,
            "transcript": self.transcript,
            "chunks": self.chunks or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "fact_check": {
                "status": self.fact_check_status or "pending",
                "timestamp": self.fact_check_timestamp.isoformat() if self.fact_check_timestamp else None,
                "worth_checking": self.worth_checking,
                "reason": self.worth_checking_reason,
                "claims": self.claims or [],
                "results": self.fact_check_results or [],
                "context": self.fact_check_context,
                "error": self.fact_check_error,
                "search_summary": self.search_summary or {},
                "total_duration": self.total_search_duration,
                "metadata": self.search_metadata or {}
            }
        }

# 添加新表：单个断言核查结果（可选，用于细粒度跟踪）
class FactCheckResult(db.Model):
    """存储单个断言的核查结果详情"""
    __tablename__ = "fact_check_results"

    id = db.Column(db.Integer, primary_key=True)
    transcript_id = db.Column(db.Integer, db.ForeignKey("video_transcripts.id"), nullable=False)
    claim = db.Column(db.Text, nullable=False)  # 断言内容
    is_true = db.Column(db.String(20), nullable=False)  # 是/否/未确定/错误
    conclusion = db.Column(db.Text)  # 结论解释
    search_duration = db.Column(db.Float)  # 搜索耗时
    search_query = db.Column(db.Text)  # 搜索查询
    search_details = db.Column(db.JSON)  # 搜索细节（keywords, grade等）
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "claim": self.claim,
            "is_true": self.is_true,
            "conclusion": self.conclusion,
            "search_duration": self.search_duration,
            "search_query": self.search_query,
            "search_details": self.search_details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
class ContentAnalysis(db.Model):
    __tablename__ = "content_analysis"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    intent = db.Column(db.JSON, default=[])  # 意图列表
    statements = db.Column(db.JSON, default=[])  # 陈述内容列表
    summary = db.Column(db.Text, nullable=True)  # 内容摘要

    # P1: 背景信息充分性评估
    p1_score = db.Column(db.Float, nullable=True)
    p1_reasoning = db.Column(db.Text, nullable=True)

    # P2: 背景信息准确性评估
    p2_score = db.Column(db.Float, nullable=True)
    p2_reasoning = db.Column(db.Text, nullable=True)

    # P3: 内容完整性评估
    p3_score = db.Column(db.Float, nullable=True)
    p3_reasoning = db.Column(db.Text, nullable=True)

    # P4: 不当意图评估
    p4_score = db.Column(db.Float, nullable=True)
    p4_reasoning = db.Column(db.Text, nullable=True)

    # P5: 发布者历史评估
    p5_score = db.Column(db.Float, nullable=True)
    p5_reasoning = db.Column(db.Text, nullable=True)

    # P6: 情感煽动性评估
    p6_score = db.Column(db.Float, nullable=True)
    p6_reasoning = db.Column(db.Text, nullable=True)

    # P7: 诱导行为评估
    p7_score = db.Column(db.Float, nullable=True)
    p7_reasoning = db.Column(db.Text, nullable=True)

    # P8: 信息一致性评估
    p8_score = db.Column(db.Float, nullable=True)
    p8_reasoning = db.Column(db.Text, nullable=True)

    risk_level = db.Column(db.String(20), nullable=True)  # 风险等级：low, medium, high
    risk_probability = db.Column(db.Float, nullable=True)  # 风险概率值
    analysis_report = db.Column(db.Text, nullable=True)  # 存储生成的分析报告
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 关联的视频
    video = db.relationship(
        "VideoFile", backref=db.backref("content_analysis", uselist=False)
    )

    def to_dict(self):
        """转换为字典，方便API返回"""
        result = {
            # 现有字段...
            "assessments": {
                "p1": {"score": self.p1_score, "reasoning": self.p1_reasoning},
                "p2": {"score": self.p2_score, "reasoning": self.p2_reasoning},
                "p3": {"score": self.p3_score, "reasoning": self.p3_reasoning},
                "p4": {"score": self.p4_score, "reasoning": self.p4_reasoning},
                "p5": {"score": self.p5_score, "reasoning": self.p5_reasoning},
                "p6": {"score": self.p6_score, "reasoning": self.p6_reasoning},
                "p7": {"score": self.p7_score, "reasoning": self.p7_reasoning},
                "p8": {"score": self.p8_score, "reasoning": self.p8_reasoning},
            },
            "risk": {"level": self.risk_level, "probability": self.risk_probability},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        # 添加报告字段（如果存在）
        if self.analysis_report:
            result["analysis_report"] = self.analysis_report

        return result

    def update_assessment(self, assessment_item, score, reasoning=None):
        """更新特定评估项的得分和理由"""
        if assessment_item == "p1":
            self.p1_score = score
            if reasoning:
                self.p1_reasoning = reasoning
        elif assessment_item == "p2":
            self.p2_score = score
            if reasoning:
                self.p2_reasoning = reasoning
        elif assessment_item == "p3":
            self.p3_score = score
            if reasoning:
                self.p3_reasoning = reasoning
        elif assessment_item == "p4":
            self.p4_score = score
            if reasoning:
                self.p4_reasoning = reasoning
        elif assessment_item == "p5":
            self.p5_score = score
            if reasoning:
                self.p5_reasoning = reasoning
        elif assessment_item == "p6":
            self.p6_score = score
            if reasoning:
                self.p6_reasoning = reasoning
        elif assessment_item == "p7":
            self.p7_score = score
            if reasoning:
                self.p7_reasoning = reasoning
        elif assessment_item == "p8":
            self.p8_score = score
            if reasoning:
                self.p8_reasoning = reasoning


# 单独视频上传任务状态跟踪表
class VideoProcessingTask(db.Model):
    __tablename__ = "video_processing_tasks"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    task_type = db.Column(
        db.String(30), nullable=False
    )  # transcription, summary, assessment
    status = db.Column(
        db.String(30), default="pending"
    )  # pending, processing, completed, failed
    progress = db.Column(db.Float, default=0.0)  # 0-100%
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error = db.Column(db.Text, nullable=True)
    attempts = db.Column(db.Integer, default=0)  # 重试次数

    # 关联
    video = db.relationship(
        "VideoFile", backref=db.backref("processing_tasks", lazy=True)
    )

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error": self.error,
            "attempts": self.attempts,
        }

class UserAnalysisTask(db.Model):
    """用于跟踪用户分析任务的表"""

    __tablename__ = "user_analysis_tasks"

    id = db.Column(db.Integer, primary_key=True)
    platform = db.Column(
        db.String(20), nullable=False
    )  # 'douyin', 'tiktok', 'bilibili' 等
    platform_user_id = db.Column(
        db.String(255), nullable=False, index=True
    )  # 平台上的用户ID
    nickname = db.Column(db.String(255))  # 用户昵称（冗余存储，方便查询）
    avatar = db.Column(db.String(500))  # 用户头像URL
    digital_human_probability = db.Column(db.Float, default=0.0, nullable=False)

    # 关联到UserProfile表（如果有详细信息）
    user_profile_id = db.Column(
        db.Integer, db.ForeignKey("user_profiles.id"), nullable=True
    )
    user_profile = db.relationship(
        "UserProfile", backref=db.backref("analysis_tasks", lazy=True)
    )

    # 任务状态
    status = db.Column(
        db.String(20), default="pending"
    )  # 'pending', 'processing', 'completed', 'failed'
    progress = db.Column(db.Float, default=0.0)  # 0-100%

    # 任务配置
    analysis_type = db.Column(
        db.String(50), default="full"
    )  # 'full', 'content_only', 'user_only'等
    max_videos = db.Column(db.Integer, default=50)  # 分析的最大视频数

    # 分析结果
    result_summary = db.Column(db.Text, nullable=True)  # 简短的分析总结
    detailed_result = db.Column(db.JSON, default={})  # 详细分析结果
    risk_level = db.Column(db.String(20), nullable=True)  # 风险等级：low, medium, high

    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 错误信息
    error = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "platform": self.platform,
            "platform_user_id": self.platform_user_id,
            "nickname": self.nickname,
            "avatar": self.avatar,
            "digital_human_probability": self.digital_human_probability,  # 返回概率值
            "user_profile_id": self.user_profile_id,
            "status": self.status,
            "progress": self.progress,
            "analysis_type": self.analysis_type,
            "max_videos": self.max_videos,
            "result_summary": self.result_summary,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error": self.error,
        }


# 添加在其他表模型之后
class DouyinVideo(db.Model):
    """抖音视频数据表"""

    __tablename__ = "douyin_videos"

    # 主键和关联
    id = db.Column(db.Integer, primary_key=True)
    aweme_id = db.Column(db.String(36), unique=True, nullable=False)  # 抖音视频ID
    user_profile_id = db.Column(
        db.Integer, db.ForeignKey("user_profiles.id"), nullable=False
    )

    # 基本信息
    desc = db.Column(db.Text, nullable=True)  # 视频描述/标题
    create_time = db.Column(db.DateTime, nullable=True)  # 发布时间
    cover_url = db.Column(db.String(500), nullable=True)  # 视频封面URL
    share_url = db.Column(db.String(500), nullable=True)  # 分享链接

    # 媒体信息
    media_type = db.Column(db.Integer, nullable=True)  # 媒体类型(视频/图集)
    video_duration = db.Column(db.Integer, default=0)  # 视频时长(秒)
    is_top = db.Column(db.Boolean, default=False)  # 是否置顶

    # 统计数据
    digg_count = db.Column(db.Integer, default=0)  # 点赞数
    comment_count = db.Column(db.Integer, default=0)  # 评论数
    collect_count = db.Column(db.Integer, default=0)  # 收藏数
    share_count = db.Column(db.Integer, default=0)  # 分享数
    play_count = db.Column(db.Integer, default=0)  # 播放数

    # 附加信息
    tags = db.Column(db.Text, nullable=True)  # 视频标签，逗号分隔
    fetched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)  # 抓取时间
    # 关联关系
    video_file_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=True)
    video_file = db.relationship("VideoFile", backref=db.backref("douyin_video", uselist=False))
    user_profile = db.relationship("UserProfile", backref=db.backref("douyin_videos", lazy=True))    
    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "aweme_id": self.aweme_id,
            "desc": self.desc,
            "create_time": self.create_time.isoformat() if self.create_time else None,
            "cover_url": self.cover_url,
            "share_url": self.share_url,
            "media_type": self.media_type,
            "video_duration": self.video_duration,
            "is_top": self.is_top,
            "statistics": {
                "digg_count": self.digg_count,
                "comment_count": self.comment_count,
                "collect_count": self.collect_count,
                "share_count": self.share_count,
                "play_count": self.play_count,
            },
            "tags": self.tags.split(",") if self.tags else [],
            "fetched_at": self.fetched_at.isoformat(),
            "video_file_id": self.video_file_id
        }
# 新增数字人检测结果表
class DigitalHumanDetection(db.Model):
    """数字人检测结果表"""
    __tablename__ = "digital_human_detections"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    
    # 总体检测状态
    status = db.Column(db.String(20), default="pending")  # pending, processing, completed, failed
    
    # 新增：各个模块的独立状态
    face_status = db.Column(db.String(20), default="not_started")  # not_started, processing, completed, failed
    body_status = db.Column(db.String(20), default="not_started")
    overall_status = db.Column(db.String(20), default="not_started")
    comprehensive_status = db.Column(db.String(20), default="not_started")
    
    # 面部检测结果
    face_ai_probability = db.Column(db.Float, nullable=True)
    face_human_probability = db.Column(db.Float, nullable=True)
    face_confidence = db.Column(db.Float, nullable=True)
    face_prediction = db.Column(db.String(20), nullable=True)  # AI-Generated, Human
    face_raw_results = db.Column(db.JSON, nullable=True)
    face_started_at = db.Column(db.DateTime, nullable=True)  # 面部检测开始时间
    face_completed_at = db.Column(db.DateTime, nullable=True)  # 面部检测完成时间
    face_error_message = db.Column(db.Text, nullable=True)  # 面部检测错误信息
    
    # 躯体检测结果
    body_ai_probability = db.Column(db.Float, nullable=True)
    body_human_probability = db.Column(db.Float, nullable=True)
    body_confidence = db.Column(db.Float, nullable=True)
    body_prediction = db.Column(db.String(20), nullable=True)
    body_raw_results = db.Column(db.JSON, nullable=True)
    body_started_at = db.Column(db.DateTime, nullable=True)  # 躯体检测开始时间
    body_completed_at = db.Column(db.DateTime, nullable=True)  # 躯体检测完成时间
    body_error_message = db.Column(db.Text, nullable=True)  # 躯体检测错误信息
    
    # 整体检测结果
    overall_ai_probability = db.Column(db.Float, nullable=True)
    overall_human_probability = db.Column(db.Float, nullable=True)
    overall_confidence = db.Column(db.Float, nullable=True)
    overall_prediction = db.Column(db.String(20), nullable=True)
    overall_raw_results = db.Column(db.JSON, nullable=True)
    overall_started_at = db.Column(db.DateTime, nullable=True)  # 整体检测开始时间
    overall_completed_at = db.Column(db.DateTime, nullable=True)  # 整体检测完成时间
    overall_error_message = db.Column(db.Text, nullable=True)  # 整体检测错误信息
    
    # 综合评估结果
    comprehensive_ai_probability = db.Column(db.Float, nullable=True)
    comprehensive_human_probability = db.Column(db.Float, nullable=True)
    comprehensive_confidence = db.Column(db.Float, nullable=True)
    comprehensive_prediction = db.Column(db.String(20), nullable=True)
    comprehensive_consensus = db.Column(db.Boolean, nullable=True)
    comprehensive_votes = db.Column(db.JSON, nullable=True)  # {"ai": 2, "human": 1}
    comprehensive_started_at = db.Column(db.DateTime, nullable=True)  # 综合评估开始时间
    comprehensive_completed_at = db.Column(db.DateTime, nullable=True)  # 综合评估完成时间
    comprehensive_error_message = db.Column(db.Text, nullable=True)  # 综合评估错误信息
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    progress = db.Column(db.Integer, default=0)  # 进度百分比 0-100
    started_at = db.Column(db.DateTime, nullable=True)  # 开始时间
    current_step = db.Column(db.String(50), nullable=True)  # 当前步骤
    # 错误信息
    error_message = db.Column(db.Text, nullable=True)
    
    # 关联视频
    video = db.relationship("VideoFile", backref=db.backref("digital_human_detection", uselist=False))
    
    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "video_id": self.video_id,
            "status": self.status,
            "module_statuses": {
                "face": self.face_status,
                "body": self.body_status,
                "overall": self.overall_status,
                "comprehensive": self.comprehensive_status
            },
            "face": {
                "ai_probability": self.face_ai_probability,
                "human_probability": self.face_human_probability,
                "confidence": self.face_confidence,
                "prediction": self.face_prediction,
                "raw_results": self.face_raw_results,
                "status": self.face_status,
                "started_at": self.face_started_at.isoformat() if self.face_started_at else None,
                "completed_at": self.face_completed_at.isoformat() if self.face_completed_at else None,
                "error_message": self.face_error_message
            } if self.face_ai_probability is not None or self.face_status != "not_started" else None,
            "body": {
                "ai_probability": self.body_ai_probability,
                "human_probability": self.body_human_probability,
                "confidence": self.body_confidence,
                "prediction": self.body_prediction,
                "raw_results": self.body_raw_results,
                "status": self.body_status,
                "started_at": self.body_started_at.isoformat() if self.body_started_at else None,
                "completed_at": self.body_completed_at.isoformat() if self.body_completed_at else None,
                "error_message": self.body_error_message
            } if self.body_ai_probability is not None or self.body_status != "not_started" else None,
            "overall": {
                "ai_probability": self.overall_ai_probability,
                "human_probability": self.overall_human_probability,
                "confidence": self.overall_confidence,
                "prediction": self.overall_prediction,
                "raw_results": self.overall_raw_results,
                "status": self.overall_status,
                "started_at": self.overall_started_at.isoformat() if self.overall_started_at else None,
                "completed_at": self.overall_completed_at.isoformat() if self.overall_completed_at else None,
                "error_message": self.overall_error_message
            } if self.overall_ai_probability is not None or self.overall_status != "not_started" else None,
            "comprehensive": {
                "ai_probability": self.comprehensive_ai_probability,
                "human_probability": self.comprehensive_human_probability,
                "confidence": self.comprehensive_confidence,
                "prediction": self.comprehensive_prediction,
                "consensus": self.comprehensive_consensus,
                "votes": self.comprehensive_votes,
                "status": self.comprehensive_status,
                "started_at": self.comprehensive_started_at.isoformat() if self.comprehensive_started_at else None,
                "completed_at": self.comprehensive_completed_at.isoformat() if self.comprehensive_completed_at else None,
                "error_message": self.comprehensive_error_message
            } if self.comprehensive_ai_probability is not None or self.comprehensive_status != "not_started" else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }
class ProcessingLog(db.Model):
    __tablename__ = "processing_logs"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    task_id = db.Column(db.Integer, db.ForeignKey("video_processing_tasks.id"), nullable=True)
    task_type = db.Column(db.String(30), nullable=True)  # 冗余存储任务类型，方便查询
    level = db.Column(db.String(20), default="INFO")  # INFO, WARNING, ERROR等
    message = db.Column(db.Text, nullable=False)  # 日志内容
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    # 关联
    video = db.relationship("VideoFile", backref=db.backref("processing_logs", lazy=True))
    task = db.relationship("VideoProcessingTask", backref=db.backref("logs", lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "level": self.level,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

def init_dataset(app):
    # 对密码进行URL编码（处理特殊字符）
    encoded_password = quote_plus(app.config.PASSWORD)

    # 动态配置数据库连接
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"mysql+pymysql://{app.config.USER}:{encoded_password}@"
        f"{app.config.HOST}:{app.config.PORT}/{app.config.DB_NAME}?"
        f"charset=utf8mb4"
    )

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 280,
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
    }

    db.init_app(app)

    try:
        with app.app_context():
            db.create_all()
        print("✅ 数据库连接成功")
    except Exception as e:
        print(f"❌ 数据库连接失败: {str(e)}")
