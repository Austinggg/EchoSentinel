from flask import Blueprint
from flask_jwt_extended import get_jwt_identity, jwt_required

from utils.HttpResponse import HttpResponse

bp = Blueprint("user", __name__)


# 获取用户信息
@bp.route("/api/user/info", methods=["GET", "POST"])
@jwt_required()
def user_info():
    return HttpResponse.success(
        data={
            "username": get_jwt_identity(),
        },
        message="Request successful",
    )
