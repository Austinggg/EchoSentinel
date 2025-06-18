import os
import sys

# 设置安全环境
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FAISS_DISABLE_GPU'] = '1'
os.environ['MALLOC_CHECK_'] = '0'
os.environ['MALLOC_PERTURB_'] = '0'

print("� 开始逐步导入测试...")

try:
    print("1. 测试 Flask...")
    from flask import Flask
    print("   ✅ Flask 导入成功")
    
    print("2. 测试 utils.extensions...")
    from utils.extensions import app
    print("   ✅ utils.extensions 导入成功")
    
    print("3. 测试数据库模块...")
    from utils.database import init_dataset
    print("   ✅ 数据库模块导入成功")
    
    print("4. 测试Redis模块...")
    from utils.redis_client import init_redis
    print("   ✅ Redis模块导入成功")
    
    print("5. 测试基础API模块...")
    from api import auth
    print("   ✅ auth 导入成功")
    
    from api import menu
    print("   ✅ menu 导入成功")
    
    from api import user
    print("   ✅ user 导入成功")
    
    from api import userAnalyse
    print("   ✅ userAnalyse 导入成功")
    
    from api import videoUpload
    print("   ✅ videoUpload 导入成功")
    
    print("✅ 所有基础模块导入成功！")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()