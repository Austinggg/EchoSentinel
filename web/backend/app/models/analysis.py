import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Boolean, Text, Float, JSON
from app.utils.extensions import db


class VideoTranscript(db.Model):
    __tablename__ = "video_transcripts"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(36), db.ForeignKey("video_files.id"), nullable=False)
    transcript = db.Column(db.Text, nullable=True)
    chunks = db.Column(db.JSON, default=[])
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 事实核查相关字段
    fact_check_status = db.Column(db.String(20), default="pending")
    fact_check_timestamp = db.Column(db.DateTime, nullable=True)
    worth_checking = db.Column(db.Boolean, nullable=True)
    worth_checking_reason = db.Column(db.Text, nullable=True)
    claims = db.Column(db.JSON, default=[])
    fact_check_results = db.Column(db.JSON, default=[])
    fact_check_context = db.Column(db.Text, nullable=True)
    fact_check_error = db.Column(db.Text, nullable=True)
    search_summary = db.Column(db.JSON, default={})
    total_search_duration = db.Column(db.Float, nullable=True)
    search_metadata = db.Column(db.JSON, default={})
    
    # 关联的视频
    video = db.relationship("VideoFile", backref=db.backref("transcript_data", uselist=False))

    def to_dict(self):
        base_dict = {
            "id": self.id,
            "video_id": self.video_id,
            "transcript": self.transcript,
            "chunks": self.chunks,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "fact_check": {
                "status": self.fact_check_status,
                "timestamp": self.fact_check_timestamp.isoformat() if self.fact_check_timestamp else None,
                "worth_checking": self.worth_checking,
                "worth_checking_reason": self.worth_checking_reason,
                "claims": self.claims,
                "results": self.fact_check_results,
                "context": self.fact_check_context,
                "error": self.fact_check_error,
                "search_summary": self.search_summary,
                "total_duration": self.total_search_duration,
                "metadata": self.search_metadata
            }
        }
        return base_dict


class FactCheckResult(db.Model):
    """存储单个断言的核查结果详情"""
    __tablename__ = "fact_check_results"

    id = db.Column(db.Integer, primary_key=True)
    transcript_id = db.Column(db.Integer, db.ForeignKey("video_transcripts.id"), nullable=False)
    claim = db.Column(db.Text, nullable=False)
    is_true = db.Column(db.String(20), nullable=False)
    conclusion = db.Column(db.Text)
    search_duration = db.Column(db.Float)
    search_query = db.Column(db.Text)
    search_details = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
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
    intent = db.Column(db.JSON, default=[])
    statements = db.Column(db.JSON, default=[])
    summary = db.Column(db.Text, nullable=True)

    # P1-P8 评估字段
    p1_score = db.Column(db.Float, nullable=True)
    p1_reasoning = db.Column(db.Text, nullable=True)
    p2_score = db.Column(db.Float, nullable=True)
    p2_reasoning = db.Column(db.Text, nullable=True)
    p3_score = db.Column(db.Float, nullable=True)
    p3_reasoning = db.Column(db.Text, nullable=True)
    p4_score = db.Column(db.Float, nullable=True)
    p4_reasoning = db.Column(db.Text, nullable=True)
    p5_score = db.Column(db.Float, nullable=True)
    p5_reasoning = db.Column(db.Text, nullable=True)
    p6_score = db.Column(db.Float, nullable=True)
    p6_reasoning = db.Column(db.Text, nullable=True)
    p7_score = db.Column(db.Float, nullable=True)
    p7_reasoning = db.Column(db.Text, nullable=True)
    p8_score = db.Column(db.Float, nullable=True)
    p8_reasoning = db.Column(db.Text, nullable=True)

    risk_level = db.Column(db.String(20), nullable=True)
    risk_probability = db.Column(db.Float, nullable=True)
    analysis_report = db.Column(db.Text, nullable=True)
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 关联的视频
    video = db.relationship("VideoFile", backref=db.backref("content_analysis", uselist=False))

    def to_dict(self):
        result = {
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