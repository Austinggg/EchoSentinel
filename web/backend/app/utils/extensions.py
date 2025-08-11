from datetime import timedelta

from dynaconf import FlaskDynaconf
from flask import Flask
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

app = Flask(__name__, static_folder="dist", static_url_path="")

# 配置
FlaskDynaconf(app, settings_files=["settings.toml", ".secrets.toml"])

# JWT
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)


# 数据库
class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
