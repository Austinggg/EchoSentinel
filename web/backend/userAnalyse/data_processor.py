import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    数据处理类，支持训练和推理时的数据转换
    对外接口:
    - fit_transform(): 训练时使用，拟合并转换数据，返回DataFrame
    - transform(): 推理时使用，转换新数据，返回DataFrame
    - save(): 保存处理器状态
    - load(): 加载处理器状态
    """

    def __init__(self):
        self.preprocessor = None
        self.feature_columns = None
        self.province_categories = None
        self._initialize_columns()

    def _initialize_columns(self):
        """初始化各类列名"""
        self.bool_columns = [
            "show_favorite_list",
            "is_gov_media_vip",
            "is_mix_user",
            "is_star",
            "is_series_user",
        ]
        self.skewed_columns = [
            "aweme_count",
            "follower_count",
            "following_count",
            "total_favorited",
            "favoriting_count",
        ]
        self.columns_to_drop = [
            "city",
            "country",
            "user_age",
            "ip_province",
            "ip_location",
        ]
        self.categorical_columns = ["gender", "province"]

    def _ensure_dataframe(self, data):
        """确保输入是DataFrame，支持单条数据"""
        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, (list, tuple)) and all(isinstance(x, dict) for x in data):
            return pd.DataFrame(data)
        raise ValueError("输入必须是DataFrame、字典或字典列表")

    def _clean_data(self, df):
        """数据清洗"""
        df = df.copy()
        # 删除冗余列
        df = df.drop(columns=self.columns_to_drop, errors="ignore")

        # 处理布尔型列
        for col in self.bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    def _feature_engineering(self, df):
        """特征工程：生成ip_match特征"""
        df = df.copy()

        # 初始化ip_match为默认值0
        df["ip_match"] = 0

        # 只有当ip_location和province都存在时才计算匹配
        if "ip_location" in df.columns and "province" in df.columns:
            # 处理空值
            df["ip_location"] = df["ip_location"].fillna("未知")
            df["province"] = df["province"].fillna("未知")

            # 提取ip_province
            df["ip_province"] = df["ip_location"].apply(
                lambda x: x.split("：")[-1].strip() if "：" in x else x.strip()
            )

            # 计算匹配结果
            df["ip_match"] = df.apply(
                lambda row: int(
                    str(row["province"]).strip() == str(row["ip_province"]).strip()
                ),
                axis=1,
            )

        return df

    def _encode_categoricals(self, df):
        """分类变量编码"""
        df = df.copy()

        # 省份编码
        if "province" in df.columns:
            # 首先处理未知值
            df["province"] = (
                df["province"]
                .fillna("未知")
                .apply(lambda x: "未知" if str(x).strip() == "" else x)
            )

            if self.province_categories is not None:  # 推理模式
                df["province"] = df["province"].apply(
                    lambda x: x if x in self.province_categories else "其他"
                )
            else:  # 训练模式
                province_counts = df["province"].value_counts()
                self.province_categories = province_counts[
                    (province_counts > 100) & (~province_counts.index.isin(["未知"]))
                ].index.tolist()
                df["province"] = df["province"].apply(
                    lambda x: x if x in self.province_categories else "其他"
                )

        # 独热编码
        for col in self.categorical_columns:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)

        return df

    def _build_preprocessor(self, df):
        """构建预处理管道"""
        # 创建自定义的FunctionTransformer，支持特征名输出
        log_transformer = FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1,
            feature_names_out="one-to-one",
        )

        preprocessor = ColumnTransformer(
            [
                (
                    "skewed_features",
                    Pipeline(
                        steps=[
                            ("log_transform", log_transformer),
                            ("robust_scaler", RobustScaler()),
                        ]
                    ),
                    self.skewed_columns,
                )
            ],
            remainder="passthrough",  # ip_match会自动保留
            verbose_feature_names_out=False,
        )

        preprocessor.fit(df)
        self.feature_columns = preprocessor.get_feature_names_out()
        return preprocessor

    def _align_features(self, df):
        """对齐特征列"""
        if self.feature_columns is None:
            raise ValueError("处理器尚未训练，请先调用fit或fit_transform")

        # 添加缺失列
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # 确保列顺序一致
        return df[self.feature_columns]

    def fit(self, X, y=None):
        """拟合处理器"""
        df = self._ensure_dataframe(X)
        df = self._feature_engineering(df)  # 先进行特征工程
        df = self._encode_categoricals(df)
        df = self._clean_data(df)  # 最后清理数据
        self.preprocessor = self._build_preprocessor(df)
        return self

    def fit_transform(self, X, y=None):
        """
        训练时使用：拟合并转换数据
        参数:
            X: DataFrame或字典列表
        返回:
            处理后的DataFrame
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        推理时使用：转换新数据
        参数:
            X: DataFrame, 字典或字典列表
        返回:
            处理后的DataFrame
        """
        if self.preprocessor is None or self.feature_columns is None:
            raise ValueError("处理器尚未训练，请先调用fit或fit_transform")

        df = self._ensure_dataframe(X)
        df = self._feature_engineering(df)  # 先进行特征工程
        df = self._clean_data(df)
        df = self._encode_categoricals(df)
        df = self._align_features(df)

        # 转换为DataFrame并保留列名
        processed_data = self.preprocessor.transform(df)
        return pd.DataFrame(processed_data, columns=self.feature_columns)

    def save(self, filepath):
        """保存处理器状态"""
        processor_state = {
            "preprocessor": self.preprocessor,
            "feature_columns": self.feature_columns,
            "province_categories": self.province_categories,
            "bool_columns": self.bool_columns,
            "skewed_columns": self.skewed_columns,
            "columns_to_drop": self.columns_to_drop,
            "categorical_columns": self.categorical_columns,
        }
        joblib.dump(processor_state, filepath)

    @classmethod
    def load(cls, filepath):
        """加载处理器状态"""
        processor_state = joblib.load(filepath)
        processor = cls()
        processor.preprocessor = processor_state["preprocessor"]
        processor.feature_columns = processor_state["feature_columns"]
        processor.province_categories = processor_state["province_categories"]
        processor.bool_columns = processor_state["bool_columns"]
        processor.skewed_columns = processor_state["skewed_columns"]
        processor.columns_to_drop = processor_state["columns_to_drop"]
        processor.categorical_columns = processor_state["categorical_columns"]
        return processor
