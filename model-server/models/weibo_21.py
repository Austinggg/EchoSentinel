import json
import torch
import logging
import os
import numpy as np
from torch.utils.data import Dataset
from pymongo import MongoClient

# 配置日志
logger = logging.getLogger(__name__)

class WeiboDataset(Dataset):
    def __init__(self, mongo_uri=None, db_name=None, collection_name=None, folder_path=None, selected_features=None):
        """微博21数据集加载类
        
        参数:
            mongo_uri: MongoDB连接URI
            db_name: 数据库名称
            collection_name: 集合名称
            folder_path: 本地数据文件夹路径（备选数据源）
            selected_features: 要选择的特征列表，默认为P1-P8
        """
        # 设置默认特征（如果未指定则使用P1-P8）
        self.selected_features = selected_features if selected_features else [f"P{i}" for i in range(1, 9)]
        records = []
        
        # 从MongoDB读取数据
        if mongo_uri and db_name and collection_name:
            try:
                client = MongoClient(mongo_uri)
                db = client[db_name]
                collection = db[collection_name]
                
                # 查询文档
                cursor = collection.find({})
                for doc in cursor:
                    records.append(doc)
                
                logger.info(f"从MongoDB加载了 {len(records)} 条记录")
                client.close()
            except Exception as e:
                logger.error(f"连接MongoDB出错: {str(e)}")
                return
        # 从本地文件加载数据（保留原功能作为备选）
        elif folder_path:
            for filename in os.listdir(folder_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding="utf-8") as f:
                        data = json.load(f)
                        records.append(data)
        else:
            raise ValueError("必须提供MongoDB连接信息或本地数据文件夹路径")
            
        self.features = []
        self.labels = []
        
        for item in records:
            try:
                # 提取P1-P8并处理新的列表格式
                p_values = []
                for feature_key in self.selected_features:
                    val = 0.0  # 默认值
                    general = item.get('general', [])
                    
                    # 处理新的列表格式
                    if isinstance(general, list):
                        for entry in general:
                            if isinstance(entry, dict) and entry.get('key') == feature_key:
                                val = entry.get('value', 0.0)
                                break
                    # 向后兼容旧格式（字典格式）
                    elif isinstance(general, dict):
                        val = general.get(feature_key, 0.0)
                    
                    # 处理空值和类型转换
                    if val == "" or val is None:
                        val = 0.0
                    elif isinstance(val, str):
                        try:
                            val = float(val)
                        except ValueError:
                            val = 0.0
                    
                    p_values.append(float(val))
                
                self.features.append(p_values)
                
                # 标签处理
                label_val = item.get('label', 0)
                label = 1 if str(label_val).strip() in ("1", "true", "True") else 0
                self.labels.append(label)
                
            except (KeyError, ValueError) as e:
                print(f"数据 {item.get('_id', '未知')} 异常: {str(e)}")
                continue
                
        self.num_features = len(self.selected_features)
        # 新增诊断信息
        logger.info(f"\n数据集诊断信息:")
        logger.info(f"总样本数: {len(self.labels)}")
        if len(self.labels) > 0:
            logger.info(f"正样本比例: {sum(self.labels)/len(self.labels):.2%}")
            logger.info(f"特征维度: {len(self.features[0]) if self.features else 0}")
            logger.info(f"示例特征: {self.features[0] if self.features else []}")
            logger.info(f"对应标签: {self.labels[0] if self.labels else []}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        one_hot = torch.zeros(2)
        one_hot[label] = 1.0
        return torch.FloatTensor(self.features[idx]), one_hot