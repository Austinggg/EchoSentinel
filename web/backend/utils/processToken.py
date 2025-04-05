from datetime import timedelta  # 推荐使用时区感知时间

from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    verify_jwt_in_request,
)

from utils.extensions import app

# 过期时间
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)


# 获取access token
def generate_token(username):
    """
    生成access token
    :param username 用户名
    return token
    """
    access_token = create_access_token(identity=username)
    return access_token


def verify_token(token: str):
    """
    验证JWT令牌有效性
    :param token: JWT令牌字符串
    :return: 是否验证通过
    """
    try:
        verify_jwt_in_request(token)
        return True
    except Exception as e:
        print(e)


def get_username_from_token(token: str):
    """
    从JWT令牌提取用户名（不验证令牌有效性）
    :param token: JWT令牌字符串
    :return: 用户名
    """
    current_user = get_jwt_identity(token)
    return current_user
