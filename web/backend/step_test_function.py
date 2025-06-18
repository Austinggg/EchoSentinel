import os
import sys

# 设置安全环境
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FAISS_DISABLE_GPU'] = '1'
os.environ['MALLOC_CHECK_'] = '0'
os.environ['MALLOC_PERTURB_'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("🔍 分步测试 function.py 导入...")

try:
    print("1. 测试基础Python模块...")
    import hashlib
    import json
    from pathlib import Path
    from typing import Annotated, Tuple
    print("   ✅ 基础Python模块导入成功")
    
    print("2. 测试 numpy...")
    import numpy as np
    print("   ✅ numpy 导入成功")
    
    print("3. 测试 pandas...")
    import pandas as pd
    print("   ✅ pandas 导入成功")
    
    print("4. 测试 torch...")
    import torch
    print("   ✅ torch 导入成功")
    
    print("5. 测试 sklearn.manifold.TSNE...")
    from sklearn.manifold import TSNE
    print("   ✅ sklearn.manifold.TSNE 导入成功")
    
    print("6. 测试 sqlalchemy...")
    from sqlalchemy import select
    print("   ✅ sqlalchemy 导入成功")
    
    print("7. 测试数据库模块...")
    from utils.database import UserProfile
    from utils.extensions import db
    print("   ✅ 数据库模块导入成功")
    
    print("8. 测试 userAnalyse.AEModel...")
    from userAnalyse.AEModel import infer_8features, infer_loss
    print("   ✅ userAnalyse.AEModel 导入成功")
    
    print("9. 测试 userAnalyse.CoverFeature...")
    from userAnalyse.CoverFeature import ImageFeatureExtractor
    print("   ✅ userAnalyse.CoverFeature 导入成功")
    
    print("10. 测试 userAnalyse.data_processor...")
    from userAnalyse.data_processor import DataProcessor
    print("   ✅ userAnalyse.data_processor 导入成功")
    
    print("11. 测试 userAnalyse.OLSH...")
    from userAnalyse.OLSH import OLsh
    print("   ✅ userAnalyse.OLSH 导入成功")
    
    print("✅ 所有模块导入成功")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()