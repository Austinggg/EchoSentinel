import os
from pathlib import Path

from flask import Blueprint, request, send_from_directory
from sqlalchemy import select

from userAnalyse.function import cal_loss as userAnalyse_main
from utils.database import UserProfile, db
from utils.HttpResponse import HttpResponse

bp = Blueprint("userAnalyse", __name__)


@bp.route("/api/userAnalyse/demo", methods=["GET"])
def userAnalyse_demo():
    """返回用户分析demo数据"""
    return HttpResponse.success(
        data=[
            {
                "sec_uid": "MS4wLjABAAAALxPAZM7qgk5yrE_L-Qu4eZW_L2MJ-ApSH6yXNdoOShU",
                "nickname": "陈三岁（恐怖游戏）",
                "tag": "正常",
            },
            {
                "sec_uid": "MS4wLjABAAAA2p-25_YoJpsOCl5UZYTEBd8ES8eYk2K-Uqr-rb0PShUGcBi0NGT-I0TllKcPVril",
                "nickname": "大鹅头",
                "tag": "正常",
            },
            {
                "sec_uid": "MS4wLjABAAAAqtBQr20xuJwSjZvCd5Nq2mM8_ysxD-14rMsXSbj-ygE",
                "nickname": "萧枭同学",
                "tag": "正常",
            },
            {
                "sec_uid": "MS4wLjABAAAAEwah8sDthIJ7AJlINAC594-bDJ-R5BeeftswlRlPVOs",
                "nickname": "潮绘师王大",
                "tag": "正常",
            },
            {
                "sec_uid": "MS4wLjABAAAAK_HOplkAQxvmihLndIRCpHv1FAWn7vuidIoHEkhDaiMfjgt87ELed3ZzKlRhkXu_",
                "nickname": "空帆🎶",
                "tag": "正常",
            },
            {
                "sec_uid": "MS4wLjABAAAAK_HOplkAQxvmihLndIRCpHv1FAWn7vuidIoHEkhDaiMfjgt87ELed3ZzKlRhkXu_",
                "nickname": "风险用户示例",
                "tag": "风险",
            },
        ]
    )


@bp.route("/api/userAnalyse/getProfile", methods=["GET", "POST"])
def userAnalyse_getProfile():
    """返回用户信息"""
    sec_uid = request.get_json().get("sec_uid")
    stmt = select(UserProfile).where(UserProfile.sec_uid == sec_uid)
    userProfile = db.session.execute(stmt).scalars().first()
    return HttpResponse.success(data=userProfile.to_dict())


@bp.route("/api/userAnalyse/getCover/<filename>", methods=["GET"])
def userAnalyse_getCover(filename: str):
    """返回封面图片"""
    image_directory = Path("data/userAnalyse/video_covers")
    if not os.path.isfile(os.path.join(image_directory, filename)):
        return HttpResponse.error("Image not found")
    return send_from_directory(image_directory.resolve(), filename)


@bp.post("/api/userAnalyse/getRank")
def userAnalyse_getRank():
    """返回用户分析数据"""
    sec_uid = request.get_json().get("sec_uid")
    stmt = select(UserProfile).where(UserProfile.sec_uid == sec_uid)
    userProfiles = db.session.execute(stmt).scalars().first()
    loss = userAnalyse_main(userProfiles)
    return HttpResponse.success(data={"loss": loss})
