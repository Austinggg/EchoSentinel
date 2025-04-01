from dynaconf import FlaskDynaconf
from flask import Flask, send_from_directory

from api.auth import init_auth_routes
from api.user import init_user_routes
from utils.database import User, db, init_dataset

app = Flask(__name__, static_folder="dist", static_url_path="")

FlaskDynaconf(app, settings_files=["settings.toml"])
init_dataset(app)


@app.route("/")
def index():
    return send_from_directory("dist", "index.html")


init_auth_routes(app)
init_user_routes(app)

# def hello_world():
#     return "Hello, World!"


@app.route("/adduser")
def add_user():
    username = "vben"
    password = "123456"
    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return "success"


if __name__ == "__main__":
    app.run(debug=True, port=8000)  # 开启调试模式（包含热重载）
