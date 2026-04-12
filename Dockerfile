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
# ... (Mantenha o topo igual até o pip install)
RUN pip install --cache-dir /tmp/pip-cache -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple

# 1. Primeiro, copia TUDO para o container
COPY . .

# 2. MOVE ou COPIA o conteúdo de src para a raiz (/app)
# Isso garante que train.py e infer.py fiquem em /app/train.py
RUN cp -r src/* .

# 3. Mantém o PYTHONPATH por segurança para imports internos
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# O CMD aqui é uma redundância, o servidor vai usar o startCmd dele
CMD ["python", "train.py"]