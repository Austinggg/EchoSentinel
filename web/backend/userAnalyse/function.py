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
    print(cover_features.shape, user_features.shape)
    print(infer(user_features, cover_features))
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
