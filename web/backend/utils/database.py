import json
from typing import Dict, Optional
from urllib.parse import quote_plus  # 用于密码编码

from sqlalchemy import JSON, BigInteger, Boolean, String
from sqlalchemy.orm import Mapped, mapped_column
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
