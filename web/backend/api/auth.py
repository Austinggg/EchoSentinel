import datetime
from datetime import timezone  # 推荐使用时区感知时间

import jwt
from flask import jsonify, request
from sqlalchemy import select

from utils.database import User, db
from utils.HttpResponse import ErrorHttpResponse, HttpResponse


def init_auth_routes(app):
    @app.route("/api/auth/login", methods=["GET", "POST"])
    def auth_login():
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        stmt = select(User).where(User.username == username)
        user = db.session.execute(stmt).scalars().first()
        if user:
            if user.check_password(password):
                payload = {
                    "username": user.username,
                    "exp": datetime.datetime.now(tz=timezone.utc)
                    + datetime.timedelta(hours=24),
                }
                secret_key = app.config["JWT_SECRET_KEY"]
                access_token = jwt.encode(payload, secret_key)
                result_data = {
                    "user": username,
                    "accessToken": access_token,
                }
                response = HttpResponse(
                    code=0, data=result_data, message="Request successful"
                )
                return jsonify(response.to_dict())
            else:
                return jsonify(
                    ErrorHttpResponse(
                        message="Username or password is incorrect."
                    ).to_dict()
                )
        else:
            return jsonify(ErrorHttpResponse(message="User not found.").to_dict())
