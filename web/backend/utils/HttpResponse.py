import json
from typing import TypeVar

from flask import Response, jsonify

T = TypeVar("T")


class HttpResponse(Response):
    def __init__(self, code: int = 0, data: T = None, message: str = ""):
        """
        :param code: 0 means success, others means fail
        :param data: response data
        :param message: response message
        """
        content = json.dumps({"code": code, "data": data, "message": message})
        super().__init__(response=content, content_type="application/json")

    @classmethod
    def success(cls, data, message="success"):
        return cls(code=0, data=data, message=message)

    @classmethod
    def error(cls, data, message="error"):
        return cls(code=-1, data=data, message=message)


def success_response(data=None, message="操作成功", code=200):
    """
    生成成功响应
    :param data: 响应数据
    :param message: 响应消息
    :param code: 响应状态码
    :return: JSON响应
    """
    return jsonify({
        "code": code,
        "message": message,
        "data": data
    })


def error_response(code=500, message="操作失败", data=None):
    """
    生成错误响应
    :param code: 错误状态码
    :param message: 错误消息
    :param data: 响应数据
    :return: JSON响应
    """
    return jsonify({
        "code": code,
        "message": message,
        "data": data
    }), code
