from flask import Blueprint, request
from flask_jwt_extended import create_access_token
from sqlalchemy import select

from app.models.user import User  # 按需导入User模型
from app.utils.extensions import db  # 从extensions导入db
from app.utils.extensions import app
from app.utils.HttpResponse import HttpResponse

bp = Blueprint("auth", __name__)


# 登录
@bp.route("/api/auth/login", methods=["GET", "POST"])
def auth_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    # 用户名或密码不存在
    if not username or not password:
        return HttpResponse.error(message="用户名或者密码不能为空")
    stmt = select(User).where(User.username == username)
    user = db.session.execute(stmt).scalars().first()
    # 用户不存在
    if not user:
        return HttpResponse.error(message="用户不存在")
    else:
        # 用户名或密码错误
        if not user.check_password(password):
            return HttpResponse.error(message="用户名或者密码错误")
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
        username = request_json.get("username")
        password = request_json.get("password")

        # 参数验证
        if not username or not password:
            return HttpResponse.error(message="用户名和密码不能为空")

        # 检查用户名是否已存在
        if db.session.query(User).filter_by(username=username).first():
            return HttpResponse.error(message="用户名已存在")

        # 创建新用户
        user = User()
        user.username = username
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return HttpResponse.success(data="success", message="注册成功")

    except Exception as e:
        db.session.rollback()
        app.logger.warning(str(e))
        return HttpResponse.error(message="注册失败")