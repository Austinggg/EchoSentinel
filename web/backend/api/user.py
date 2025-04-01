from flask import jsonify

from utils.HttpResponse import HttpResponse


def init_user_routes(app):
    @app.route("/api/user/info", methods=["GET", "POST"])
    def user_info():
        result_data = {
            "user": "vben",
            "age": 30,
        }
        # print(request.headers)
        # 成功响应
        response = HttpResponse(code=0, data=result_data, message="Request successful")
        return jsonify(response.to_dict())
