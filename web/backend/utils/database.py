from typing import Optional
from urllib.parse import quote_plus  # 用于密码编码

from sqlalchemy import String
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
