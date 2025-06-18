import os
import logging
import warnings

def setup_faiss_environment():
    """设置FAISS环境，禁用GPU相关功能和警告"""
    
    print("🔧 正在配置FAISS环境...")
    
    # 设置环境变量禁用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['FAISS_DISABLE_GPU'] = '1'
    
    # 禁用相关警告
    warnings.filterwarnings('ignore', category=UserWarning, module='faiss')
    warnings.filterwarnings('ignore', message='.*GPU.*')
    warnings.filterwarnings('ignore', message='.*GpuIndexIVFFlat.*')
    warnings.filterwarnings('ignore', message='.*CUDA.*')
    
    # 设置日志级别 - 只显示错误
    faiss_logger = logging.getLogger('faiss')
    faiss_logger.setLevel(logging.ERROR)
    
    faiss_loader_logger = logging.getLogger('faiss.loader')
    faiss_loader_logger.setLevel(logging.ERROR)
    
    try:
        # 尝试导入faiss并配置
        import faiss
        
        # 确保只使用CPU
        if hasattr(faiss, 'omp_set_num_threads'):
            faiss.omp_set_num_threads(4)  # 限制CPU线程数
        
        print("✅ FAISS CPU模式配置完成")
        return True
        
    except ImportError as e:
        print(f"⚠️  FAISS未安装或导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️  FAISS配置出现问题: {e}")
        return False

# 如果直接运行此文件，执行配置
if __name__ == "__main__":
    setup_faiss_environment()