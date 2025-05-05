import os

# MongoDB连接信息
MONGODB = {
    "uri": os.getenv("MONGODB_URI", "mongodb://root:CV3d!GXxZp4aApx@1.95.190.99:27017/admin"),
    "db_name": os.getenv("MONGODB_DB", "TRAING_DATA"),
    "collection_name": os.getenv("MONGODB_COLLECTION", "WEIBO_21")
}

# 模型相关配置
MODEL = {
    # 模型保存路径
    "save_dir": os.getenv("MODEL_SAVE_DIR", "./models"),
    "decision_model_path": os.getenv("DECISION_MODEL_PATH", "mod/Desision-mod/best_model.pth"),
    # 训练相关参数
    "training": {
        "num_conjuncts": 10,
        "num_epochs": 50,
        "patience": 5,
        "delta": 0.01,
        "delta_decay_delay": 100,
        "delta_decay_steps": 50,
        "delta_decay_rate": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32
    }
}

# 特征映射
FEATURE_MAP = {
    1: "信息充分性", 
    2: "信息准确性", 
    3: "内容完整性", 
    4: "意图正当性",
    5: "发布者信誉", 
    6: "情感中立性",
    7: "无诱导行为", 
    8: "信息一致性"
}