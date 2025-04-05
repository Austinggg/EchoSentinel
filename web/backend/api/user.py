from flask import Blueprint, request

from utils.HttpResponse import HttpResponse

user_bp = Blueprint("user", __name__)


# 获取用户信息
@user_bp.route("/api/user/info", methods=["GET", "POST"])
def user_info():
    data = {
        "user": "vben",
        "age": 30,
    }
    print(request.headers)
    return HttpResponse.success(data=data, message="Request successful")
