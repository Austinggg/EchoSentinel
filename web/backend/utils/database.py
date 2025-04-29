import json
import datetime
from typing import Dict, Optional, Any
from urllib.parse import quote_plus  # 用于密码编码

from sqlalchemy import JSON, BigInteger, Boolean, String, ForeignKey, Integer, DateTime
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
    
    # 添加与VideoFile的关系
    videos = relationship("VideoFile", back_populates="user")

    def to_dict(self):
        covers_data = json.loads(self.covers) if self.covers else {}
        return {
            "id": self.id,
            "sec_uid": self.sec_uid,
            "hash_sec_uid": self.hash_sec_uid,
            "nickname": self.nickname,
            "gender": self.gender,
            "city": self.city or "未知",
            "province": self.province or "未知",
            "country": self.country or "未知",
            "aweme_count": self.aweme_count,  # 1
            "follower_count": self.follower_count,  # 2
            "following_count": self.following_count,  # 3
            "total_favorited": self.total_favorited,  # 4
            "favoriting_count": self.favoriting_count,  # 5
            "user_age": self.user_age,
            "ip_location": self.ip_location,
            "show_favorite_list": self.show_favorite_list,
            "is_gov_media_vip": self.is_gov_media_vip,
            "is_mix_user": self.is_mix_user,
            "is_star": self.is_star,
            "is_series_user": self.is_series_user,
            "covers": [
                f"http://localhost:8000/api/userAnalyse/getCover/{x}.jpg"
                for x in covers_data.values()
            ],
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
    
    def to_dict(self):
        tags_list = self.tags.split(',') if self.tags else []
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
            "url": f"/api/videos/{self.id}"
        }

# 视频转录表
class VideoTranscript(db.Model):
    __tablename__ = "video_transcripts"
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    transcript = db.Column(db.Text, nullable=True)  # 完整转录文本
    chunks = db.Column(db.JSON, default=[])  # 带时间戳的分段文本
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # 关联的视频
    video = db.relationship("VideoFile", backref=db.backref("transcript_data", uselist=False))

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
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # 关联的视频
    video = db.relationship("VideoFile", backref=db.backref("content_analysis", uselist=False))
    
    def to_dict(self):
        """转换为字典，方便API返回"""
        return {
            "id": self.id,
            "video_id": self.video_id,
            "intent": self.intent,
            "statements": self.statements,
            "summary": self.summary,
            "assessments": {
                "p1": {"score": self.p1_score, "reasoning": self.p1_reasoning},
                "p2": {"score": self.p2_score, "reasoning": self.p2_reasoning},
                "p3": {"score": self.p3_score, "reasoning": self.p3_reasoning},
                "p4": {"score": self.p4_score, "reasoning": self.p4_reasoning},
                "p5": {"score": self.p5_score, "reasoning": self.p5_reasoning},
                "p6": {"score": self.p6_score, "reasoning": self.p6_reasoning},
                "p7": {"score": self.p7_score, "reasoning": self.p7_reasoning},
                "p8": {"score": self.p8_score, "reasoning": self.p8_reasoning}
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def update_assessment(self, assessment_item, score, reasoning=None):
        """更新特定评估项的得分和理由"""
        if assessment_item == "p1":
            self.p1_score = score
            if reasoning: self.p1_reasoning = reasoning
        elif assessment_item == "p2":
            self.p2_score = score
            if reasoning: self.p2_reasoning = reasoning
        elif assessment_item == "p3":
            self.p3_score = score
            if reasoning: self.p3_reasoning = reasoning
        elif assessment_item == "p4":
            self.p4_score = score
            if reasoning: self.p4_reasoning = reasoning
        elif assessment_item == "p5":
            self.p5_score = score
            if reasoning: self.p5_reasoning = reasoning
        elif assessment_item == "p6":
            self.p6_score = score
            if reasoning: self.p6_reasoning = reasoning
        elif assessment_item == "p7":
            self.p7_score = score
            if reasoning: self.p7_reasoning = reasoning
        elif assessment_item == "p8":
            self.p8_score = score
            if reasoning: self.p8_reasoning = reasoning

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