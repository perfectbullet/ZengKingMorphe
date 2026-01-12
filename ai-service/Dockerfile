# syntax=docker/dockerfile:latest

########################################
# 基础镜像：统一系统依赖
########################################
FROM ubuntu:22.04 AS base

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 使用阿里云镜像源（Ubuntu 22.04）
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list \
    && sed -i 's@security.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        build-essential \
        curl \
        locales \
        tzdata \
    && sed -i 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen zh_CN.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置时区为中国/上海
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

# 让 python / pip 指向 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 设置中文环境变量
ENV LANG=zh_CN.UTF-8
ENV LC_ALL=zh_CN.UTF-8
ENV LANGUAGE=zh_CN:zh
ENV TZ=Asia/Shanghai

# 配置 pip 源（阿里云）
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

########################################
# 依赖构建阶段：安装 requirements 并输出 wheels
########################################
FROM base AS builder

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip wheel --wheel-dir=/wheels -r requirements.txt

########################################
# 最终镜像：使用虚拟环境 + 离线安装依赖
########################################
FROM base AS runtime

COPY requirements.txt .
COPY --from=builder /wheels /wheels

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# 激活虚拟环境
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

COPY . .

# 创建必要的目录
RUN mkdir -p /app/chroma_db /app/logs

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
