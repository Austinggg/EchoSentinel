## vue-vben-admin

### 运行调试

```shell
# web/vue-vben-admin
pnpm dev:ele
```

### 修改设置

[vben 文档](https://doc.vben.pro/guide/in-depth/login.html)

```ini
#.env.development
# 是否开启 Nitro Mock服务，true 为开启，false 为关闭
VITE_NITRO_MOCK=false

# vite.config.mts
# 改为真实接口位置
target: 'http://localhost:8000/api',
```

## backend

### 运行调试

后端启动
``` shell
cd web/backend/ && source .venv/bin/activate && python3 app.py
```

前端启动
``` shell
cd web/vue-vben-admin && pnpm dev:ele
```