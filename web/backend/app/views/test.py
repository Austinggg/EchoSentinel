from flask import Blueprint, jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required
from sqlalchemy import select

from userAnalyse.function import cal_loss
from app.utils.database import User, UserProfile
from app.utils.extensions import db

bp = Blueprint("test", __name__)


@bp.route("/hello")
def test():
    return "hello"


# 测试接口
@bp.route("/adduser")
def add_user():
    username = "Admin"
    password = "123456"
    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return "success"


# 测试jwt
@bp.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    # Access the identity of the current user with get_jwt_identity
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200


@bp.get("/findLoss")
def find_loss():
    # 查询所有用户
    stmt = select(UserProfile)
    users = db.session.execute(stmt).scalars().all()

    user_loss_pairs = [(user, cal_loss(user)) for user in users]

    top_5_users = sorted(user_loss_pairs, key=lambda x: x[1], reverse=True)[:5]

    # 提取 sec_uid 和 loss
    result = [
        {
            "sec_uid": user.sec_uid,
            "loss": loss,
        }
        for user, loss in top_5_users
    ]

    return jsonify(result)
