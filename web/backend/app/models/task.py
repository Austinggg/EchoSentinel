import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, Float, JSON
from sqlalchemy.orm import relationship
from app.utils.extensions import db


class VideoProcessingTask(db.Model):
    __tablename__ = "video_processing_tasks"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    task_type = db.Column(db.String(30), nullable=False)
    status = db.Column(db.String(30), default="pending")
    progress = db.Column(db.Float, default=0.0)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error = db.Column(db.Text, nullable=True)
    attempts = db.Column(db.Integer, default=0)

    # 关联
    video = relationship("VideoFile", backref=db.backref("processing_tasks", lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "attempts": self.attempts,
        }


class UserAnalysisTask(db.Model):
    """用于跟踪用户分析任务的表"""
    __tablename__ = "user_analysis_tasks"

    id = db.Column(db.Integer, primary_key=True)
    platform = db.Column(db.String(20), nullable=False)
    platform_user_id = db.Column(db.String(255), nullable=False, index=True)
    nickname = db.Column(db.String(255))
    avatar = db.Column(db.String(500))
    digital_human_probability = db.Column(db.Float, default=0.0, nullable=False)

    # 关联到UserProfile表
    user_profile_id = db.Column(db.Integer, db.ForeignKey("user_profiles.id"), nullable=True)
    user_profile = relationship("UserProfile", backref=db.backref("analysis_tasks", lazy=True))

    # 任务状态
    status = db.Column(db.String(20), default="pending")
    progress = db.Column(db.Float, default=0.0)

    # 任务配置
    analysis_type = db.Column(db.String(50), default="full")
    max_videos = db.Column(db.Integer, default=50)

    # 分析结果
    result_summary = db.Column(db.Text, nullable=True)
    detailed_result = db.Column(db.JSON, default={})
    risk_level = db.Column(db.String(20), nullable=True)

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
            "digital_human_probability": self.digital_human_probability,
            "user_profile_id": self.user_profile_id,
            "status": self.status,
            "progress": self.progress,
            "analysis_type": self.analysis_type,
            "max_videos": self.max_videos,
            "result_summary": self.result_summary,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error": self.error,
        }


class ProcessingLog(db.Model):
    __tablename__ = "processing_logs"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    task_id = db.Column(db.Integer, db.ForeignKey("video_processing_tasks.id"), nullable=True)
    task_type = db.Column(db.String(30), nullable=True)
    level = db.Column(db.String(20), default="INFO")
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    # 关联
    video = relationship("VideoFile", backref=db.backref("processing_logs", lazy=True))
    task = relationship("VideoProcessingTask", backref=db.backref("logs", lazy=True))

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
