from typing import Any, Dict, TypeVar

T = TypeVar("T")


class HttpResponse:
    def __init__(self, code: int = 0, data: T = None, message: str = ""):
        """
        :param code: 0 means success, others means fail
        :param data: response data
        :param message: response message
        """
        self.code = code
        self.data = data
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "data": self.data, "message": self.message}


class ErrorHttpResponse(HttpResponse):
    def __init__(self, code=-1, message="error"):
        self.code = code
        self.data = None
        self.message = message
