#!/bin/bash

# 设置模型路径
MODELS_PATH='/root/lizhenyu/LLM_repo/models'

mkdir -p "$MODELS_PATH"

# 切换到模型路径
cd "$MODELS_PATH" || exit 1  # 加上退出机制以防路径不存在

# 设置ASR目录变量
RAG_PATH='rag'

# 创建ASR目录
mkdir -p "./$RAG_PATH"  # 使用 -p 参数以递归创建目录

# 激活conda环境
conda activate agent

# 下载Embedding模型
modelscope download --model maidalun/bce-embedding-base_v1 --cache_dir "./$RAG_PATH"

# 下载Reranker模型
modelscope download --model maidalun/bce-reranker-base_v1 --cache_dir "./$RAG_PATH"
