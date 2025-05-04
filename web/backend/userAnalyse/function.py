import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from userAnalyse.AEModel import infer
from userAnalyse.CoverFeature import ImageFeatureExtractor
from userAnalyse.data_processor import DataProcessor
from userAnalyse.OLSH import OLsh
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
) -> float:
    """
    输入:
        test_loss: 测试样本的重构损失
        stats: 训练阶段计算的统计量
    输出:
        异常分数（范围10-80），变化平缓，仅明显异常时高分
    """
    # 1. 计算标准化损失（基于均值+标准差）
    normalized_loss = (test_loss - stats["mean"]) / stats["std"]

    # 2. 使用Sigmoid函数（调整参数使曲线更平缓）
    # 参数说明：
    # - 除以3降低敏感度（原z-3改为z/3-1）
    # - 分子用70保证最大值≈80
    sigmoid_score = 70 / (1 + np.exp(-(normalized_loss / 2 - 1)))

    # 3. 映射到10-80分范围
    final_score = 10 + sigmoid_score

    # 4. 确保分数不超出范围（理论上不需要，但防止极端情况）
    return np.clip(final_score, 10, 80)


def plot_data():
    csv_file = Path("userAnalyse/output8.csv")
    output_path = csv_file.parent / "tsne_with_labels.txt"
    index_file = Path("userAnalyse/olsh_index.joblib")

    df = pd.read_csv(csv_file)
    feature_columns = [f"feature_{i}" for i in range(8)]
    hash_sec_uids = df["hash_sec_uid"].values
    original_data = df[feature_columns].values

    if output_path.exists():
        combined_data = np.loadtxt(output_path, delimiter=",", dtype=object)
        return combined_data

    # --- Step 2: Get Cluster Labels using OLsh ---
    olsh = OLsh()  # Initialize OLsh
    olsh.load(index_file)

    cluster_labels = olsh.labels  # Shape (n_samples,)

    # --- Step 3: Perform t-SNE (if combined data wasn't loaded) ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_data = tsne.fit_transform(original_data)

    # --- Step 4: Combine t-SNE results and Labels ---
    labels_column = cluster_labels.reshape(-1, 1)
    uids_column = hash_sec_uids.reshape(-1, 1)
    combined_data = np.hstack((reduced_data, labels_column, uids_column))

    # --- Step 5: Save combined data ---
    column_formats = ["%.4f", "%.4f", "%d", "%s"]
    np.savetxt(output_path, combined_data, delimiter=",", fmt=column_formats)

    # --- Step 6: Return combined data ---
    return combined_data


def get_feature_by_uid(sec_uid: str):
    """根据id获取特征向量"""
    hash_sec_uid = hashlib.md5(sec_uid.encode()).hexdigest()
    df = pd.read_csv("userAnalyse/output8.csv")
    row = df[df["hash_sec_uid"] == hash_sec_uid]
    features = row[[f"feature_{i}" for i in range(8)]].values[0]
    return features
