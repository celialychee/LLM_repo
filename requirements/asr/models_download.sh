#!/bin/bash

# 设置模型路径
MODELS_PATH='/root/lizhenyu/LLM_repo/models'

mkdir -p "$MODELS_PATH"

# 切换到模型路径
cd "$MODELS_PATH" || exit 1  # 加上退出机制以防路径不存在

# 设置ASR目录变量
ASR_PATH='asr'

# 创建ASR目录
mkdir -p "./$ASR_PATH"  # 使用 -p 参数以递归创建目录

# 激活conda环境
conda activate agent

# 下载语音识别模型（Paraformer）
modelscope download --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --cache_dir "./$ASR_PATH"

# 下载语音端点检测模型（FSMN）
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --cache_dir "./$ASR_PATH"

# 下载加标点模型（CT-Transformer）
modelscope download --model iic/punc_ct-transformer_cn-en-common-vocab471067-large --cache_dir "./$ASR_PATH"