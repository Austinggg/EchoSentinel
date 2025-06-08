#!/bin/bash
nohup \
    /root/EchoSentinel/model-server/.venv/bin/python \
    /root/EchoSentinel/model-server/app.py \
    >app.log 2>&1 &

echo "flask 服务已启动 PID: $!"
echo "日志记录见 app.log"
