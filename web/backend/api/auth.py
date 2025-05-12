from flask import Blueprint, request
from flask_jwt_extended import create_access_token
from sqlalchemy import select

from utils.database import User, db
from utils.extensions import app
from utils.HttpResponse import HttpResponse

bp = Blueprint("auth", __name__)


# 登录
@bp.route("/api/auth/login", methods=["GET", "POST"])
def auth_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    # 用户名或密码不存在
    if not username or not password:
        return HttpResponse.error(message="Username and password are required")
    stmt = select(User).where(User.username == username)
    user = db.session.execute(stmt).scalars().first()
    # 用户不存在
    if not user:
        return HttpResponse.error(message="User not found.")
    else:
        # 用户名或密码错误
        if not user.check_password(password):
            return HttpResponse.error(message="Username or password is incorrect.")
        else:
            return HttpResponse.success(
                data={
                    "user": username,
                    "accessToken": create_access_token(identity=username),
                },
                message="Request successful",
            )


# 登出
@bp.route("/api/auth/logout", methods=["GET", "POST"])
def auth_logout():
    return HttpResponse.success("logout")


@bp.post("/api/auth/register")
def auth_register():
    try:
        request_json = request.get_json()
        user = User()
        user.username = request_json.get("username")
        user.set_password(request_json.get("password"))
        db.session.add(user)
        db.session.commit()
        return HttpResponse.success(data="success")
    except Exception as e:
        app.logger.warning(str(e))
        return HttpResponse.error(e)
