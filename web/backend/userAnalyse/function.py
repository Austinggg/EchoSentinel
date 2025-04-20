import json

import numpy as np
import pandas as pd
import torch

from userAnalyse.AEModel import infer
from userAnalyse.CoverFeature import ImageFeatureExtractor
from userAnalyse.data_processor import DataProcessor
from utils.database import UserProfile

columns_order = [
    "sec_uid",
    "gender",
    "city",
    "province",
    "country",
    "aweme_count",
    "follower_count",
    "following_count",
    "total_favorited",
    "favoriting_count",
    "user_age",
    "ip_location",
    "show_favorite_list",
    "is_gov_media_vip",
    "is_mix_user",
    "is_star",
    "is_series_user",
]


def cal_loss(userProfile):
    covers = json.loads(userProfile.covers).values()
    covers = [x for x in covers if x is not None and not isinstance(x, float)]
    features = extract_cover_features(covers)
    cover_features = torch.stack(features).mean(dim=0)

    df = userProfile_dataFrame(userProfile)
    df = preprocess(df)
    # print(df)
    df = df.drop(columns=["sec_uid"])
    df = df.astype(np.float32).values
    user_features = torch.tensor(df)
    # print(cover_features.shape, user_features.shape)
    # print(infer(user_features, cover_features))
    return infer(user_features, cover_features)


def userProfile_dataFrame(userProfile: UserProfile):
    pd.set_option("display.max_columns", None)  # 显示所有列
    data_for_df = {col: [getattr(userProfile, col, None)] for col in columns_order}
    df = pd.DataFrame(data_for_df, columns=columns_order)
    # print(df)
    return df


def preprocess(df):
    processor = DataProcessor.load("userAnalyse/data_processor.pkl")
    processed_test = processor.transform(df)
    return processed_test
    # print(processed_test)


def extract_cover_features(covers):
    extractor = ImageFeatureExtractor()
    cover_path = [f"data/userAnalyse/video_covers/{x}.jpg" for x in covers]
    cover_features = [extractor(extractor.preprocess(x)) for x in cover_path]
    return cover_features


def get_anomaly_score(
    test_loss: float,
    stats: dict = {
        "mean": np.float64(0.012980108857162363),
        "std": np.float64(0.016118451871877056),
        "q50": np.float64(0.005857843905687332),
        "q99": np.float64(0.0701417756080627),
        "max": np.float64(0.12325551360845566),
    },
    method: str = "hybrid",
) -> float:
    """
    输入:
        test_loss: 测试样本的重构损失
        stats: 训练阶段保存的统计量字典
    输出:
        0-100的异常分数
    """
    if method == "zscore":
        z = (test_loss - stats["mean"]) / stats["std"]
        return 100 / (1 + np.exp(-(z - 3)))  # Sigmoid映射

    elif method == "quantile":
        if test_loss <= stats["q50"]:
            return 0.0
        elif test_loss >= stats["q99"]:
            return 100.0
        else:
            return 100 * (test_loss - stats["q50"]) / (stats["q99"] - stats["q50"])

    elif method == "dynamic":
        normalized = (test_loss - stats["mean"]) / (stats["max"] - stats["mean"])
        return 100 * np.clip(normalized, 0, 1) ** 2

    elif method == "hybrid":  # 推荐方法
        z = (test_loss - stats["mean"]) / stats["std"]
        if z <= 0:
            return 0.0
        else:
            linear_score = min(
                100, 100 * (test_loss - stats["q50"]) / (stats["q99"] - stats["q50"])
            )
            sigmoid_score = 100 / (1 + np.exp(-(z - 3)))
            return max(linear_score, sigmoid_score)

    else:
        raise ValueError("Method must be 'zscore', 'quantile', 'dynamic', or 'hybrid'")
