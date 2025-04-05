import json
from typing import TypeVar

from flask import Response

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
