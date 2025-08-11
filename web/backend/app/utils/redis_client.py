import json
import logging
import redis
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis客户端封装类"""
    
    def __init__(self, host: str, port: int, password: str, db: int = 0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        self._test_connection()
    
    def _test_connection(self):
        """测试Redis连接"""
        try:
            self.redis_client.ping()
            logger.info("✅ Redis连接成功")
        except Exception as e:
            logger.error(f"❌ Redis连接失败: {str(e)}")
            raise
    
    def set_task_status(self, task_key: str, status_data: Dict[str, Any], expire_time: int = 3600) -> bool:
        """设置任务状态"""
        try:
            serialized_data = json.dumps(status_data, ensure_ascii=False)
            result = self.redis_client.setex(task_key, expire_time, serialized_data)
            return bool(result)
        except Exception as e:
            logger.error(f"设置任务状态失败 - Key: {task_key}, Error: {str(e)}")
            return False
    
    def get_task_status(self, task_key: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        try:
            data = self.redis_client.get(task_key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"获取任务状态失败 - Key: {task_key}, Error: {str(e)}")
            return None
    
    def update_task_progress(self, task_key: str, progress: float, current_step: str = None) -> bool:
        """更新任务进度"""
        try:
            # 获取现有状态
            current_status = self.get_task_status(task_key)
            if not current_status:
                return False
            
            # 更新进度
            current_status['progress'] = progress
            current_status['updated_at'] = json.dumps({"$date": {"$numberLong": str(int(__import__('time').time() * 1000))}})
            
            if current_step:
                current_status['current_step'] = current_step
            
            return self.set_task_status(task_key, current_status)
        except Exception as e:
            logger.error(f"更新任务进度失败 - Key: {task_key}, Error: {str(e)}")
            return False
    
    def delete_task_status(self, task_key: str) -> bool:
        """删除任务状态"""
        try:
            result = self.redis_client.delete(task_key)
            return bool(result)
        except Exception as e:
            logger.error(f"删除任务状态失败 - Key: {task_key}, Error: {str(e)}")
            return False
    
    def publish_task_update(self, channel: str, message: Dict[str, Any]) -> bool:
        """发布任务更新消息"""
        try:
            serialized_message = json.dumps(message, ensure_ascii=False)
            result = self.redis_client.publish(channel, serialized_message)
            return bool(result)
        except Exception as e:
            logger.error(f"发布消息失败 - Channel: {channel}, Error: {str(e)}")
            return False
    
    def get_all_task_keys(self, pattern: str = "task:*") -> list:
        """获取所有匹配的任务键"""
        try:
            return self.redis_client.keys(pattern)
        except Exception as e:
            logger.error(f"获取任务键失败 - Pattern: {pattern}, Error: {str(e)}")
            return []
    
    def set_cache(self, key: str, value: Any, expire_time: int = 3600) -> bool:
        """设置缓存"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            result = self.redis_client.setex(key, expire_time, value)
            return bool(result)
        except Exception as e:
            logger.error(f"设置缓存失败 - Key: {key}, Error: {str(e)}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            data = self.redis_client.get(key)
            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            return None
        except Exception as e:
            logger.error(f"获取缓存失败 - Key: {key}, Error: {str(e)}")
            return None
    
    def clear_expired_tasks(self) -> int:
        """清理过期的任务状态"""
        try:
            task_keys = self.get_all_task_keys()
            cleared_count = 0
            
            for key in task_keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -2:  # 键不存在
                    cleared_count += 1
                elif ttl == -1:  # 键存在但没有过期时间，设置默认过期时间
                    self.redis_client.expire(key, 3600)
            
            return cleared_count
        except Exception as e:
            logger.error(f"清理过期任务失败: {str(e)}")
            return 0

# Redis装饰器，用于自动处理Redis连接错误
def redis_error_handler(fallback_return=None):
    """Redis错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except redis.ConnectionError:
                logger.warning(f"Redis连接失败，{func.__name__} 操作跳过")
                return fallback_return
            except Exception as e:
                logger.error(f"Redis操作异常 - {func.__name__}: {str(e)}")
                return fallback_return
        return wrapper
    return decorator

# 全局Redis客户端实例
redis_client: Optional[RedisClient] = None

def init_redis(app):
    """初始化Redis客户端"""
    global redis_client
    try:
        redis_client = RedisClient(
            host=app.config.REDIS_HOST,
            port=app.config.REDIS_PORT,
            password=app.config.REDIS_PASSWORD,
            db=app.config.REDIS_DB
        )
        logger.info("Redis初始化成功")
    except Exception as e:
        logger.error(f"Redis初始化失败: {str(e)}")
        redis_client = None

def get_redis_client() -> Optional[RedisClient]:
    """获取Redis客户端实例"""
    return redis_client