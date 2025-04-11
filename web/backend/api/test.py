from flask import Blueprint, jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required

from utils.database import User
from utils.extensions import db

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
