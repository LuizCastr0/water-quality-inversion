# 使用比赛指定的镜像
FROM 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-py310.19

# 设置容器内的当前工作目录
WORKDIR /app

# 接收构建参数（由 GitLab CI 自动处理）
ARG PIP_CACHE_DIR

# 首先复制依赖文件以利用 Docker 缓存
COPY requirements.txt .

# 安装依赖
# 使用华为云镜像源以加快在中国服务器上的下载速度
RUN pip install --cache-dir /tmp/pip-cache -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple

# 复制项目的所有文件到容器的 /app 目录下
COPY . .

# --- 关键修正部分 ---

# 1. 设置环境变量，让 Python 能够识别 src 文件夹中的模块
# 这样即使你在 /app 下运行，也能正确执行 'import dataset' 等操作
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
ENV PYTHONUNBUFFERED=1

# 2. 修正启动命令：
# 删除了 Flask 相关配置，改为直接运行你的训练脚本
# 由于你的文件在 src/ 下，路径必须写成 src/train.py
CMD ["python", "src/train.py"]