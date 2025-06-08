import datetime
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import cv2
from flask import Blueprint, Flask, request
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from extensions import app, db
from model_api_call import DigitalHumanEvaluator

bp = Blueprint("aigc detection", __name__)


class VideoFile(db.Model):
    __tablename__ = "video_files"

    id = mapped_column(String(36), primary_key=True)
    filename = mapped_column(String(255), nullable=False)
    extension = mapped_column(String(10))  # 只存储文件扩展名
    size = mapped_column(Integer)
    mime_type = mapped_column(String(100))
    upload_time = mapped_column(DateTime, default=datetime.datetime.utcnow)
    user_id = mapped_column(Integer, ForeignKey("user_profiles.id"), nullable=True)

    # 分析结果相关字段
    summary = mapped_column(String(500), default="处理中")
    risk_level = mapped_column(String(20), default="processing")
    status = mapped_column(String(50), default="processing")

    # 新增爬虫数据字段
    publish_time = mapped_column(DateTime, nullable=True)  # 视频发布时间
    tags = mapped_column(String(500), nullable=True)  # 视频标签，以逗号分隔

    # 关联的用户
    user = relationship("UserProfile", back_populates="videos")
    # 添加视频来源相关字段
    source_url = mapped_column(String(500), nullable=True)  # 源视频URL
    source_platform = mapped_column(
        String(50), nullable=True
    )  # 源平台（douyin/tiktok）
    source_id = mapped_column(String(100), nullable=True)  # 平台上的原始ID

    # 添加数字人检测字段
    aigc_use = mapped_column(String(50), nullable=True)
    aigc_face = mapped_column(String(50), nullable=True)
    aigc_body = mapped_column(String(50), nullable=True)
    aigc_whole = mapped_column(String(50), nullable=True)
    digital_human_probability = mapped_column(db.Float, default=0.0, nullable=False)


class UserProfile(db.Model):
    __tablename__ = "user_profiles"  # 表名

    id: Mapped[Optional[int]] = mapped_column(primary_key=True)
    sec_uid: Mapped[str] = mapped_column(String(255), index=True)
    hash_sec_uid: Mapped[str] = mapped_column(String(255))
    nickname: Mapped[str] = mapped_column(String(255))
    signature: Mapped[Optional[str]] = mapped_column(String(500))  # 添加用户签名字段
    gender: Mapped[Optional[str]] = mapped_column(String(50))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    province: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    aweme_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    follower_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    following_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    total_favorited: Mapped[Optional[int]] = mapped_column(BigInteger)
    favoriting_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    user_age: Mapped[Optional[int]] = mapped_column(BigInteger)
    ip_location: Mapped[Optional[str]] = mapped_column(String(100))
    show_favorite_list: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_gov_media_vip: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_mix_user: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_star: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_series_user: Mapped[Optional[bool]] = mapped_column(Boolean)
    covers: Mapped[Dict] = mapped_column(JSON, default={})
    avatar_medium: Mapped[Optional[str]] = mapped_column(String(255))

    # 添加与VideoFile的关系
    videos = relationship("VideoFile", back_populates="user")


def start_face(app: Flask):
    """执行python命令并将输出重定向到文件"""
    log_file = "aigc_detection/log/face.log"

    try:
        with open(log_file, "w") as f:
            subprocess.run(
                [
                    "/root/EchoSentinel/model-server/.venv/bin/python",
                    "/root/EchoSentinel/model-server/mod/DeepfakeBench/run_pipeline.py",
                ],
                check=True,
                stdout=f,  # 标准输出到文件
                stderr=subprocess.STDOUT,  # 错误输出合并到标准输出
                text=True,
            )
        print(f"命令执行完成，输出已保存到 {log_file}")
        app.logger.info("将face数据写入数据库")
        write_face_data(app)
    except subprocess.CalledProcessError as e:
        with open(log_file, "a") as f:
            f.write(f"\n命令执行失败: {e.stderr}\n")
        print(f"命令执行失败，详情见 {log_file}")


def write_face_data(app: Flask):
    log_path = Path(__file__).parent / "mod" / "DeepfakeBench" / "results" / "xception"
    logs = list(log_path.glob("*.json"))
    removed = []
    with app.app_context():
        for log in logs:
            data = {}
            with open(log, "r", encoding="utf-8") as f:
                data = json.load(f)

            video_id = data.get("UADFV").get("video_name")
            if video_id.endswith(".mp4"):
                video_id = video_id[:-4]  # 去掉最后4个字符(.mp4)
            pred_mean = data.get("UADFV").get("pred_mean")
            face = "fake" if pred_mean > 0.5 else "real"
            stmt = select(VideoFile).where(VideoFile.id == video_id)
            try:
                video = db.session.execute(stmt).scalars().first()
                if video:
                    video.aigc_face = face
                    removed.append(video_id)
                else:
                    log.unlink()
                    removed.append(video_id)
            except Exception as e:
                app.logger.error(f"数据库操作出错: {e}")
                db.session.rollback()

        db.session.commit()
    clean_face_data(removed)
    return None


def clean_face_data(unexist: list):
    dataset_path = (
        Path(__file__).parent
        / "mod"
        / "DeepfakeBench"
        / "datasets"
        / "rgb"
        / "UADFV"
        / "fake"
    )
    frames_path = dataset_path / "frames"
    landmarks = dataset_path / "landmarks"
    for item in unexist:
        p1 = dataset_path / str(item)
        p2 = landmarks / str(item)
        p3 = frames_path / str(item)
        if p1.exists():
            shutil.rmtree(p1)
        if p2.exists():
            shutil.rmtree(p2)
        if p3.exists():
            shutil.rmtree(p3)


def start_body(app: Flask, video_id: str):
    image_path = Path(__file__).parent / "aigc_detection" / video_id / f"{video_id}.jpg"

    evaluator = DigitalHumanEvaluator()
    evaluator.evaluate_image(
        str(image_path),
        output_dir="whole_evaluation_results",
    )
    write_body_data(app)


def extract_middle_frame(video_path, output_image_path):
    """提取视频中间的一帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_image_path, frame)

    cap.release()
    return ret


def write_body_data(app: Flask):
    result_path = Path(__file__).parent / "whole_evaluation_results"
    result_files = result_path.glob("*.json")
    result_files = list(result_files)
    with app.app_context():
        for file in result_files:
            data = {}
            with open(file, "r") as f:
                data = json.load(f)
            video_id: str = data.get("image_path")
            total_score = data.get("total_score")
            video_id = video_id.split("/")[-2]
            #
            stmt = select(VideoFile).where(VideoFile.id == video_id)

            video = db.session.execute(stmt).scalars().first()
            if video:
                video.aigc_body = total_score
        db.session.commit()

    return f"{video_id},{total_score}"


def start_whole(app, video_id):
    log_file = f"aigc_detection/log/whole-{video_id}.log"
    try:
        with open(log_file, "w") as f:
            subprocess.run(
                [
                    "/opt/conda/envs/dire/bin/python",
                    "/root/EchoSentinel/model-server/DIRE/demo.py",
                    "-f",
                    f"/root/EchoSentinel/model-server/aigc_detection/{video_id}/{video_id}.jpg",
                    "-m",
                    "/root/EchoSentinel/model-server/DIRE/lsun_adm.pth",
                ],
                check=True,
                stdout=f,  # 标准输出到文件
                stderr=subprocess.STDOUT,  # 错误输出合并到标准输出
                text=True,
                cwd="/root/EchoSentinel/model-server/DIRE",
            )
        print(f"命令执行完成，输出已保存到 {log_file}")
        app.logger.info("将face数据写入数据库")
        write_whole_data(app, video_id)
    except subprocess.CalledProcessError as e:
        with open(log_file, "a") as f:
            f.write(f"\n命令执行失败: {e.stderr}\n")
        print(f"命令执行失败，详情见 {log_file}")
    pass


def write_whole_data(app, video_id):
    result_path = Path(__file__).parent / "aigc_detection"
    result_file = result_path / video_id / "whole.json"
    with app.app_context():
        data = {}
        with open(result_file, "r") as f:
            data = json.load(f)

        prob = data.get("prob")

        #
        stmt = select(VideoFile).where(VideoFile.id == video_id)

        video = db.session.execute(stmt).scalars().first()
        if video:
            video.aigc_whole = prob
        db.session.commit()

    return f"{video_id},{prob}"


@bp.get("/aigc-detection-service/test")
def aigc_detection_service_test():
    return "s"


@bp.post("/aigc-detection-service/startProcess")
def aigc_detection_service_start_process():
    video_id = request.args.get("video_id")
    file = request.files["file"]
    file_path = (
        Path(__file__).parent
        / "mod"
        / "DeepfakeBench"
        / "datasets"
        / "rgb"
        / "UADFV"
        / "fake"
    )

    # 保存视频并开始处理
    file_path = file_path / f"{video_id}"
    file_path.mkdir(parents=True, exist_ok=True)

    app.logger.info(f"上传文件信息：file.filename:{file.filename}")
    file.save(file_path / f"{file.filename}")

    file.seek(0)
    aigc_path = Path(__file__).parent / "aigc_detection" / str(video_id)

    aigc_path.mkdir(parents=True, exist_ok=True)
    file.save(aigc_path / f"{file.filename}")

    extract_middle_frame(aigc_path / f"{file.filename}", aigc_path / f"{video_id}.jpg")
    # 面部
    # try:
    #     process_face_thread = threading.Thread(
    #         target=start_face,
    #         args=(app,),
    #         daemon=True,
    #     )
    #     process_face_thread.start()
    # except Exception as e:
    #     app.logger.warning(e)
    # 躯干
    # try:
    #     process_whole_thread = threading.Thread(
    #         target=start_body,
    #         args=(
    #             app,
    #             video_id,
    #         ),
    #         daemon=True,
    #     )
    #     process_whole_thread.start()
    # except Exception as e:
    #     app.logger.warning(e)
    # 整体
    # try:
    #     process_whole_thread = threading.Thread(
    #         target=start_whole,
    #         args=(
    #             app,
    #             video_id,
    #         ),
    #         daemon=True,
    #     )
    #     process_whole_thread.start()
    # except Exception as e:
    #     app.logger.warning(e)
    return f"start process: {video_id} {file.filename}"
