from flask import Blueprint
from sqlalchemy import select

from utils.database import VideoFile, db
from utils.HttpResponse import HttpResponse

bp = Blueprint("aigc-detection", __name__)


@bp.get("/api/aigc-detection/tableData")
def aigc_tableData():
    try:
        stmt = select(VideoFile.aigc_face, VideoFile.aigc_body).where(
            VideoFile.aigc_use == "yes"
        )
        results = db.session.execute(stmt).all()
        tableData = [{"face": row[0], "body": row[1], "whole": None} for row in results]
        return HttpResponse.success(data={"tableData": tableData})
    except Exception:
        return HttpResponse.error(message="查询失败")
