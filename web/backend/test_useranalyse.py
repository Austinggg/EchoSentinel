import os
import sys

# 设置安全环境
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FAISS_DISABLE_GPU'] = '1'
os.environ['MALLOC_CHECK_'] = '0'
os.environ['MALLOC_PERTURB_'] = '0'

print("🔍 测试 userAnalyse 模块...")

try:
    print("1. 测试基础模块...")
    from flask import Flask
    from utils.extensions import app
    print("   ✅ 基础模块正常")
    
    print("2. 测试 userAnalyse 导入...")
    import api.userAnalyse
    print("   ✅ userAnalyse 模块导入成功")
    
    print("3. 测试获取蓝图...")
    from api.userAnalyse import user_analyse_api
    print("   ✅ userAnalyse 蓝图获取成功")
    
    print("✅ userAnalyse 模块测试通过")
    
except Exception as e:
    print(f"❌ userAnalyse 模块测试失败: {e}")
    import traceback
    traceback.print_exc()