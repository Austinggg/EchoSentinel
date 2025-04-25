import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from flask import Blueprint, request, send_from_directory
from sqlalchemy import select

from userAnalyse.function import cal_loss, get_anomaly_score, plot_data
from userAnalyse.OLSH import OLsh, find_most_similar_cluster, find_most_similar_user
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
            # {
            #     "sec_uid": "MS4wLjABAAAAK_HOplkAQxvmihLndIRCpHv1FAWn7vuidIoHEkhDaiMfjgt87ELed3ZzKlRhkXu_",
            #     "nickname": "空帆🎶",
            #     "tag": "正常",
            # },
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
    loss = cal_loss(userProfiles)
    return HttpResponse.success(
        data={
            "loss": round(loss, 4),
            "anomaly_score": round(get_anomaly_score(loss), 2),
        }
    )


@bp.route("/api/userAnalyse/similarCluster", methods=["GET", "POST"])
def userAnalyse_similarCluster():
    """相似集群"""
    sec_uid = request.get_json().get("sec_uid")
    hash_sec_uid = hashlib.md5(sec_uid.encode()).hexdigest()
    print(hash_sec_uid)
    olsh = OLsh(min_samples=20)
    olsh.load("userAnalyse/olsh_index.joblib")
    df = pd.read_csv("userAnalyse/output8.csv")
    row = df[df["hash_sec_uid"] == hash_sec_uid]
    features = row[[f"feature_{i}" for i in range(8)]].values[0]

    similarClusterIndex = find_most_similar_cluster(features, olsh)
    with open("userAnalyse/olsh_index.json", "r", encoding="utf-8") as f:
        clusters = json.load(f)
    similarCluster = {
        f"cluster_{i}": clusters.get(f"cluster_{i}")[:10] for i in similarClusterIndex
    }

    all_hashes = [h for v in similarCluster.values() for h in v]
    stmt = select(UserProfile).where(UserProfile.hash_sec_uid.in_(all_hashes))
    users = {u.hash_sec_uid: u for u in db.session.execute(stmt).scalars()}

    result_clusters = [
        {
            "cluster_id": cluster_id,
            "avatar_list": [users[h].avatar_medium for h in hash_list],
        }
        for cluster_id, hash_list in similarCluster.items()
    ]
    return HttpResponse.success(data={"similarCluster": result_clusters})


@bp.route("/api/userAnalyse/similarUser", methods=["GET", "POST"])
def userAnalyse_similarUser():
    """相似用户"""
    sec_uid = request.get_json().get("sec_uid")
    hash_sec_uid = hashlib.md5(sec_uid.encode()).hexdigest()
    print(hash_sec_uid)
    olsh = OLsh(min_samples=20)
    olsh.load("userAnalyse/olsh_index.joblib")
    df = pd.read_csv("userAnalyse/output8.csv")
    row = df[df["hash_sec_uid"] == hash_sec_uid]
    features = row[[f"feature_{i}" for i in range(8)]].values[0]

    similarUsers = find_most_similar_user(features, olsh)
    all_hashes = [item[0] for item in similarUsers]
    stmt = select(UserProfile).where(UserProfile.hash_sec_uid.in_(all_hashes))
    users = {u.hash_sec_uid: u for u in db.session.execute(stmt).scalars()}
    return_users = [
        {
            "hash_sec_uid": item[0],
            "similarity": item[1],
            "avatar_medium": users[item[0]].avatar_medium,
            "nickname": users[item[0]].nickname,
            "sec_uid": users[item[0]].sec_uid,
        }
        for item in similarUsers
    ]

    return HttpResponse.success(data={"similarUser": return_users})


@bp.route("/api/userAnalyse/clusterPlotData", methods=["GET"])
def userAnalyse_clusterPlotData():
    show_data = plot_data().tolist()
    uids = [x[3] for x in show_data]

    stmt = select(
        UserProfile.hash_sec_uid,
        UserProfile.nickname,
        UserProfile.sec_uid,
        UserProfile.avatar_medium,
    ).where(UserProfile.hash_sec_uid.in_(uids))
    results = db.session.execute(stmt).all()
    uid_to_info = {
        hash_uid: (sec_uid, nickname, avatar)
        for hash_uid, nickname, sec_uid, avatar in results
    }
    show_date_withurl = [x[:3] + list(uid_to_info.get(x[3])) for x in show_data]

    return HttpResponse.success(data={"data": show_date_withurl})
