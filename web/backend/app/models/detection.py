import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Boolean, Text, Float, JSON
from sqlalchemy.orm import relationship
from app.utils.extensions import db


class DigitalHumanDetection(db.Model):
    """数字人检测结果表"""
    __tablename__ = "digital_human_detections"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    
    # 总体检测状态
    status = db.Column(db.String(20), default="pending")
    
    # 新增：各个模块的独立状态
    face_status = db.Column(db.String(20), default="not_started")
    body_status = db.Column(db.String(20), default="not_started")
    overall_status = db.Column(db.String(20), default="not_started")
    comprehensive_status = db.Column(db.String(20), default="not_started")
    
    # 面部检测结果
    face_ai_probability = db.Column(db.Float, nullable=True)
    face_human_probability = db.Column(db.Float, nullable=True)
    face_confidence = db.Column(db.Float, nullable=True)
    face_prediction = db.Column(db.String(20), nullable=True)
    face_raw_results = db.Column(db.JSON, nullable=True)
    face_started_at = db.Column(db.DateTime, nullable=True)
    face_completed_at = db.Column(db.DateTime, nullable=True)
    face_error_message = db.Column(db.Text, nullable=True)
    
    # 躯体检测结果
    body_ai_probability = db.Column(db.Float, nullable=True)
    body_human_probability = db.Column(db.Float, nullable=True)
    body_confidence = db.Column(db.Float, nullable=True)
    body_prediction = db.Column(db.String(20), nullable=True)
    body_raw_results = db.Column(db.JSON, nullable=True)
    body_started_at = db.Column(db.DateTime, nullable=True)
    body_completed_at = db.Column(db.DateTime, nullable=True)
    body_error_message = db.Column(db.Text, nullable=True)
    
    # 整体检测结果
    overall_ai_probability = db.Column(db.Float, nullable=True)
    overall_human_probability = db.Column(db.Float, nullable=True)
    overall_confidence = db.Column(db.Float, nullable=True)
    overall_prediction = db.Column(db.String(20), nullable=True)
    overall_raw_results = db.Column(db.JSON, nullable=True)
    overall_started_at = db.Column(db.DateTime, nullable=True)
    overall_completed_at = db.Column(db.DateTime, nullable=True)
    overall_error_message = db.Column(db.Text, nullable=True)
    
    # 综合评估结果
    comprehensive_ai_probability = db.Column(db.Float, nullable=True)
    comprehensive_human_probability = db.Column(db.Float, nullable=True)
    comprehensive_confidence = db.Column(db.Float, nullable=True)
    comprehensive_prediction = db.Column(db.String(20), nullable=True)
    comprehensive_consensus = db.Column(db.Boolean, nullable=True)
    comprehensive_votes = db.Column(db.JSON, nullable=True)
    comprehensive_started_at = db.Column(db.DateTime, nullable=True)
    comprehensive_completed_at = db.Column(db.DateTime, nullable=True)
    comprehensive_error_message = db.Column(db.Text, nullable=True)
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    progress = db.Column(db.Integer, default=0)
    started_at = db.Column(db.DateTime, nullable=True)
    current_step = db.Column(db.String(50), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    
    # 关联视频
    video = relationship("VideoFile", backref=db.backref("digital_human_detection", uselist=False))
    
    def to_dict(self):
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
            # ...existing code for face, body, overall, comprehensive...
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }
