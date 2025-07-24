# EchoSentinel

"EchoSentinel：面向短视频平台的数字人风险感知、监测与评估系统"，主要包括：（1）面部-躯干融合的数字人视频识别模块，实现对复杂伪造数字人视频的高精度识别与真实度感知；（2）动态知识协同的内容校验模块，实现跨领域语义风险检测与分析，提升对诱导诈骗、虚假宣传等风险内容的发现能力；（3）异常账号群体识别模块，能够对数字人账号的异常行为与关联社群进行快速聚类溯源，辅助平台实现高效、精准的风险防控与治理，助力构建安全可信的数字人内容生态。

## 项目架构

```
EchoSentinel/
├── web/                    # Web 应用模块
│   ├── vue-vben-admin/     # 前端应用 (Vue3 + Vben Admin)
│   └── backend/            # 后端 API 服务 (Python)
└── model-server/           # AI 模型服务
```

## 快速开始

### Docker 环境部署

#### 单独构建镜像
模型文件暂时不行
```shell
docker build -t echosentinel-frontend ./web/vue-vben-admin/
docker build -t echosentinel-backend ./web/backend/
docker build -t echosentinel-model-server ./model-server/
```

#### Docker Compose 启动

##### 1. 先启动外部依赖服务
```shell
docker pull evil0ctal/douyin_tiktok_download_api:latest
docker run -d --name douyin-api -p 8080:80 evil0ctal/douyin_tiktok_download_api:latest
```

##### 2. 启动 KAG 服务
```shell
curl -sSL https://raw.githubusercontent.com/OpenSPG/openspg/refs/heads/master/dev/release/docker-compose-west.yml -o docker-compose-west.yml
docker compose -f docker-compose-west.yml up -d
```

##### 3. 构建并启动主要服务
```shell
docker-compose build
docker-compose up -d
```

### 源码部署

如果不使用 Docker，可以手动安装和配置环境。

#### 环境要求
- Python >= 3.10
- Node.js >= 22
- uv (Python 包管理器)
项目推荐使用 uv 作为 Python 包管理器，确保环境中已安装 uv
- pnpm (Node.js 包管理器)
前端使用 pnpm 作为包管理器，确保环境中已安装 pnpm

#### 本地开发

##### 1. 后端服务启动

```shell
# 进入后端目录
cd web/backend/

# 使用 uv 安装依赖
uv sync

# 启动虚拟环境
source .venv/bin/activate

# 启动后端服务
python3 app.py
```

##### 2. 前端服务启动

```shell
# 进入前端目录
cd web/vue-vben-admin/

# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev:ele
```

##### 3. 模型服务启动
```shell
# 进入模型服务目录
cd model-server/
# 使用 uv 安装依赖
uv sync
# 启动虚拟环境
source .venv/bin/activate
# 启动模型服务
python server.py
```
## 外部依赖配置

### 模型下载与配置

系统使用以下预训练模型，请确保模型文件已正确配置：

- **DeepfakeBench**: [GitHub - SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
- **Qwen2-VL-2B-Instruct**: Hugging Face Model Hub
- **bge-large-zh-v1.5**: Hugging Face Model Hub  
- **bge-reranker-v2-m3**: Hugging Face Model Hub
- **whisper-large-v3-turbo**: Hugging Face Model Hub

### 模型路径配置
在 `model-server/mod/` 目录下配置模型路径，确保各模型能够正常加载。


### 外部服务部署

#### Douyin_TikTok_Download_API 服务

```shell
# 拉取并启动抖音下载服务
docker pull evil0ctal/douyin_tiktok_download_api:latest
docker run -d -p 8080:80 evil0ctal/douyin_tiktok_download_api:latest
```

服务文档：[GitHub - Evil0ctal/Douyin_TikTok_Download_API](https://github.com/Evil0ctal/Douyin_TikTok_Download_API)

#### OpenSPG/KAG 服务

```shell
# 下载并启动 KAG 服务
curl -sSL https://raw.githubusercontent.com/OpenSPG/openspg/refs/heads/master/dev/release/docker-compose-west.yml -o docker-compose-west.yml
docker compose -f docker-compose-west.yml up -d
```

服务文档：[GitHub - OpenSPG/KAG](https://github.com/OpenSPG/KAG)


### 数据库配置

#### 后端配置文件

修改 `web/backend/settings.toml`：


## 服务端口说明

- 前端服务：`http://localhost:5173`
- 后端服务：`http://localhost:8000`
- 模型服务：`http://localhost:8001`
- 抖音下载服务：`http://localhost:8080`
- KAG 服务：根据 docker-compose 配置

## 注意事项

1. **数据库配置**：首次运行前请确保 MySQL 数据库已创建并配置正确
2. **Redis 配置**：确保 Redis 服务正常运行并可连接
3. **Docker 服务**：外部依赖服务需要先启动 Docker 容器
4. **模型文件**：确保所有 AI 模型文件已下载并配置正确路径
5. **网络访问**：部分模型需要访问 Hugging Face，请确保网络连接正常