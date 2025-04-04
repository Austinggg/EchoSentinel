from dynaconf import FlaskDynaconf
from flask import Flask
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

app = Flask(__name__, static_folder="dist", static_url_path="")


FlaskDynaconf(app, settings_files=["settings.toml", ".secrets.toml"])
jwt = JWTManager(app)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
