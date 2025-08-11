import json
from typing import Dict, Optional
from sqlalchemy import JSON, BigInteger, Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from werkzeug.security import check_password_hash, generate_password_hash
from app.utils.extensions import db


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
    __tablename__ = "user_profiles"

    id: Mapped[Optional[int]] = mapped_column(primary_key=True)
    sec_uid: Mapped[str] = mapped_column(String(255), index=True)
    hash_sec_uid: Mapped[str] = mapped_column(String(255))
    nickname: Mapped[str] = mapped_column(String(255))
    signature: Mapped[Optional[str]] = mapped_column(String(500))
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
    district: Mapped[Optional[str]] = mapped_column(String(100))
    custom_verify: Mapped[Optional[str]] = mapped_column(String(255))
    enterprise_verify_reason: Mapped[Optional[str]] = mapped_column(String(255))
    verification_type: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    # 添加显式构造函数
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # 关系
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
            "district": self.district or "未知",
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
            "custom_verify": self.custom_verify,
            "enterprise_verify_reason": self.enterprise_verify_reason,
            "verification_type": self.verification_type,
            "covers": [
                f"http://localhost:8000/api/userAnalyse/getCover/{x}.jpg"
                for x in covers_data.values()
            ],
            "avatar_medium": self.avatar_medium
        }
