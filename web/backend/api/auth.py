from flask import Blueprint, request
from sqlalchemy import select

from utils.database import User, db
from utils.HttpResponse import HttpResponse
from utils.processToken import generate_token

auth_bp = Blueprint("auth", __name__)


# 登录
@auth_bp.route("/api/auth/login", methods=["GET", "POST"])
def auth_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    # 用户名或密码不存在
    if not username and not password:
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
            access_token = generate_token(username)
            result_data = {
                "user": username,
                "accessToken": access_token,
            }
            response = HttpResponse.success(
                data=result_data, message="Request successful"
            )
            return response


# 登出
@auth_bp.route("/api/auth/logout", methods=["GET", "POST"])
def auth_logout():
    return HttpResponse.success("logout")
