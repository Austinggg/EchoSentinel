# EchoSentinel Backend

EchoSentinel 后端服务 - 基于 Flask 的 AI 内容检测与分析平台

## 🏗️ 项目架构

```
backend/
├── app/                    # 核心应用包
│   ├── __init__.py        # Flask 应用工厂
│   ├── services/          # 业务逻辑层
│   ├── utils/             # 工具函数
│   └── views/             # API 路由层
├── config/                # 配置文件
├── test/                  # 测试代码
├── AISearch/              # AI 搜索服务模块
├── app.py                 # 传统启动方式（兼容性）
├── run.py                 # 推荐启动方式
└── requirements files     # 依赖管理
```

## 🚀 快速开始

### 环境要求

- Python 3.12+
- Redis 服务
- MySQL 数据库
- CUDA 支持（可选，用于 AI 加速）

### 安装依赖

```bash
# 使用 uv（推荐）
uv sync
```

### 环境配置

1. 复制配置文件模板：

```bash
cp .secrets.toml.example .secrets.toml
```

2. 编辑配置文件 `.secrets.toml`：

```toml
[database]
host = "localhost"
port = 3306
user = "your_db_user"
password = "your_db_password"
name = "echo_sentinel"

[redis]
host = "localhost"
port = 6379
password = ""
db = 0

[ai]
openai_api_key = "your_openai_key"
model_path = "path/to/your/models"
```

### 启动服务

```bash
# 推荐方式：使用应用工厂模式
python run.py

# 开发模式
export FLASK_ENV=development
python run.py
```

服务将在 `http://localhost:8000` 启动

## 📚 API 文档

### 核心功能模块

#### 1. 用户认证 (`/api/auth`)

- `POST /api/auth/login` - 用户登录
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/logout` - 用户登出
- `GET /api/auth/profile` - 获取用户信息

#### 2. 视频处理 (`/api/video`)

- `POST /api/video/upload` - 视频上传
- `POST /api/video/transcribe` - 视频转录
- `POST /api/video/extract` - 内容提取与摘要
- `GET /api/video/status/{task_id}` - 获取处理状态

#### 3. 内容分析 (`/api/analysis`)

- `POST /api/analysis/assessment` - 逻辑评估
- `POST /api/analysis/decision` - 内容决策
- `GET /api/analysis/report/{analysis_id}` - 生成分析报告
- `GET /api/analysis/analytics` - 获取分析统计

#### 4. 第三方平台 (`/api/platform`)

- `POST /api/platform/douyin/download` - 抖音视频下载
- `POST /api/platform/digital-human/detect` - 数字人检测

#### 5. AI 搜索 (`/api/search`)

- `POST /api/search/query` - 智能搜索
- `GET /api/search/history` - 搜索历史

#### 6. 系统管理 (`/api/system`)

- `GET /api/system/status` - 系统状态
- `POST /api/system/settings` - 系统设置

### 响应格式

成功响应：

```json
{
  "status": "success",
  "data": {...},
  "message": "操作成功"
}
```

错误响应：

```json
{
  "status": "error",
  "error_code": "ERROR_CODE",
  "message": "错误描述",
  "details": {...}
}
```

## 🔧 开发指南

### 项目结构说明

#### `/app` - 核心应用包

- **应用工厂模式**：使用 `create_app()` 函数创建 Flask 实例
- **蓝图组织**：按功能模块组织路由
- **分层架构**：视图层 → 服务层 → 数据层

#### `/app/views` - API 路由层

```python
# 示例：视频上传 API
@video_bp.route('/upload', methods=['POST'])
def upload_video():
    # 1. 参数验证
    # 2. 调用服务层
    # 3. 返回响应
    pass
```

#### `/app/services` - 业务逻辑层

```python
# 示例：视频处理服务
class VideoService:
    def process_video(self, video_data):
        # 具体的业务逻辑实现
        pass
```

#### `/app/utils` - 工具函数

- `database.py` - 数据库工具
- `redis_client.py` - Redis 客户端
- `decorators.py` - 自定义装饰器
- `validators.py` - 数据验证器

### 添加新功能

1. **创建服务类**：

```python
# app/services/new_service.py
class NewService:
    def __init__(self):
        pass

    def process_data(self, data):
        # 业务逻辑
        return result
```

2. **创建 API 路由**：

```python
# app/views/new_api.py
from flask import Blueprint
from app.services.new_service import NewService

new_bp = Blueprint('new_api', __name__)

@new_bp.route('/endpoint', methods=['POST'])
def new_endpoint():
    service = NewService()
    result = service.process_data(request.json)
    return jsonify(result)
```

3. **注册蓝图**：

```python
# app/__init__.py
def register_blueprints(app):
    from app.views.new_api import new_bp
    app.register_blueprint(new_bp, url_prefix='/api/new')
```

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest test/

# 运行特定模块测试
python -m pytest test/test_video.py

# 生成覆盖率报告
python -m pytest --cov=app test/
```

### 测试结构

```
test/
├── conftest.py           # pytest 配置和 fixtures
├── test_auth.py          # 认证模块测试
├── test_video.py         # 视频模块测试
├── test_analysis.py      # 分析模块测试
└── fixtures/             # 测试数据
```

## 🐳 部署

### Docker 部署

```bash
# 构建镜像
docker build -t echo-sentinel-backend .

# 运行容器
docker run -d \
  --name echo-sentinel \
  -p 8000:8000 \
  -v $(pwd)/.secrets.toml:/app/.secrets.toml \
  echo-sentinel-backend
```

### 生产环境配置

1. **配置环境变量**：

```bash
export FLASK_ENV=production
export DATABASE_URL=postgresql://user:pass@host:5432/dbname
export REDIS_URL=redis://host:6379/0
```

2. **使用 Gunicorn**：

```bash
gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app()"
```

## 📊 监控与日志

### 日志配置

日志文件位置：`logs/`

- `app.log` - 应用日志
- `error.log` - 错误日志
- `access.log` - 访问日志

### 性能监控

- **Redis 监控**：使用 `redis-cli monitor` 查看缓存访问
- **数据库监控**：查看慢查询日志
- **API 性能**：内置请求时间记录

## 🔒 安全考虑

- **API 认证**：基于 JWT Token 认证
- **数据验证**：输入参数严格验证
- **SQL 注入防护**：使用 ORM 参数化查询
- **XSS 防护**：输出内容转义
- **CORS 配置**：跨域请求控制

### 代码规范

- **PEP 8**：Python 代码风格
- **类型提示**：使用 typing 模块
- **文档字符串**：每个函数都需要文档
- **单元测试**：新功能必须包含测试

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

---

