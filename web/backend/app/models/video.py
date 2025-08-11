import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import mapped_column, relationship
from app.utils.extensions import db


class VideoFile(db.Model):
    __tablename__ = "video_files"

    id = mapped_column(String(36), primary_key=True)
    filename = mapped_column(String(255), nullable=False)
    extension = mapped_column(String(10))
    size = mapped_column(Integer)
    mime_type = mapped_column(String(100))
    upload_time = mapped_column(DateTime, default=datetime.datetime.utcnow)
    user_id = mapped_column(Integer, ForeignKey("user_profiles.id"), nullable=True)

    # 分析结果相关字段
    summary = mapped_column(String(500), default="处理中")
    risk_level = mapped_column(String(20), default="processing")
    status = mapped_column(String(50), default="processing")

    # 新增爬虫数据字段
    publish_time = mapped_column(DateTime, nullable=True)
    tags = mapped_column(String(500), nullable=True)

    # 关联的用户
    user = relationship("UserProfile", back_populates="videos")
    
    # 添加视频来源相关字段
    source_url = mapped_column(String(500), nullable=True)
    source_platform = mapped_column(String(50), nullable=True)
    source_id = mapped_column(String(100), nullable=True)

    # 添加数字人检测字段
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


class DouyinVideo(db.Model):
    """抖音视频数据表"""
    __tablename__ = "douyin_videos"

    id = db.Column(db.Integer, primary_key=True)
    aweme_id = db.Column(db.String(36), unique=True, nullable=False)
    user_profile_id = db.Column(db.Integer, db.ForeignKey("user_profiles.id"), nullable=False)

    # 基本信息
    desc = db.Column(db.Text, nullable=True)
    create_time = db.Column(db.DateTime, nullable=True)
    cover_url = db.Column(db.String(500), nullable=True)
    share_url = db.Column(db.String(500), nullable=True)

    # 媒体信息
    media_type = db.Column(db.Integer, nullable=True)
    video_duration = db.Column(db.Integer, default=0)
    is_top = db.Column(db.Boolean, default=False)

    # 统计数据
    digg_count = db.Column(db.Integer, default=0)
    comment_count = db.Column(db.Integer, default=0)
    collect_count = db.Column(db.Integer, default=0)
    share_count = db.Column(db.Integer, default=0)
    play_count = db.Column(db.Integer, default=0)

    # 附加信息
    tags = db.Column(db.Text, nullable=True)
    fetched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # 关联关系
    video_file_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=True)
    video_file = relationship("VideoFile", backref=db.backref("douyin_video", uselist=False))
    user_profile = relationship("UserProfile", backref=db.backref("douyin_videos", lazy=True))

    def to_dict(self):
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
