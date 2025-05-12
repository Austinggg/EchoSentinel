from flask import Blueprint
from sqlalchemy import select

from utils.database import VideoFile, db
from utils.HttpResponse import HttpResponse

bp = Blueprint("aigc-detection", __name__)


@bp.get("/api/aigc-detection/tableData")
def aigc_tableData():
    try:
        stmt = select(
            VideoFile.id, VideoFile.aigc_face, VideoFile.aigc_body, VideoFile.aigc_whole
        ).where(VideoFile.aigc_use == "yes")
        results = db.session.execute(stmt).all()
        tableData = [
            {"id": row[0], "face": row[1], "body": row[2], "whole": row[3]}
            for row in results
        ]
        return HttpResponse.success(data=tableData)
    except Exception:
        return HttpResponse.error(message="查询失败")
