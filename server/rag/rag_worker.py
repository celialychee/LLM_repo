import shutil
from pathlib import Path

import torch
import yaml

from ..web_configs import WEB_CONFIGS
from .feature_store import gen_vector_db
from .retriever import CacheRetriever

# 基础配置
CONTEXT_MAX_LENGTH = 3000  # 上下文最大长度
GENERATE_TEMPLATE = "相关背景：“{}”\n 用户的问题：“{}” \n 请阅读相关背景解答用户的问题。"  # RAG prompt 模板


def build_rag_prompt(rag_retriever: CacheRetriever, prompt):

    real_retriever = rag_retriever.get(fs_id="default")

    if isinstance(real_retriever, tuple):
        print(f" @@@ GOT real_retriever == tuple : {real_retriever}")
        return ""

    chunk, db_context, references = real_retriever.query(
        f"{prompt}", context_max_length=CONTEXT_MAX_LENGTH - 2 * len(GENERATE_TEMPLATE)
    )
    print(f"db_context = {db_context}")
    
    # 如果检索到内容就加上相关上下文
    if db_context is not None and len(db_context) > 1:
        prompt_rag = GENERATE_TEMPLATE.format(db_context, prompt)
    # 没有检索到就由模型直接生成
    else:
        print("db_context get error")
        prompt_rag = prompt

    print(f"RAG reference = {references}")
    print("=" * 20)

    return prompt_rag


def init_rag_retriever(rag_config: str, db_path: str):
    torch.cuda.empty_cache()
    retriever = CacheRetriever()
    # 初始化
    retriever.get(fs_id="default", config_path=rag_config, work_dir=db_path)
    return retriever   


def gen_rag_db(force_gen=False):
    """
    生成向量数据库。

    参数:
    force_gen - 布尔值，当设置为 True 时，即使数据库已存在也会重新生成数据库。
    """

    # 检查数据库目录是否存在，如果存在且force_gen为False，则不执行生成操作
    if Path(WEB_CONFIGS.RAG_WORK_DIR).exists() and not force_gen:
        return
    
    # 如果存在且force_gen为True, 则先删除旧目录
    if force_gen and Path(WEB_CONFIGS.RAG_WORK_DIR).exists():
        shutil.rmtree(WEB_CONFIGS.RAG_WORK_DIR)

    print("Generating rag database, pls wait ...")
    # 调用函数生成向量数据库
    gen_vector_db(
        WEB_CONFIGS.RAG_CONFIG_PATH,
        WEB_CONFIGS.RAG_REPO_DIR,
        WEB_CONFIGS.RAG_WORK_DIR,
        # 重新生成向量库并生成阈值
        update_throttle=True
    )


def load_rag_model():
    # 生成 rag 数据库
    # gen_rag_db(True)
    gen_rag_db()
    # 加载 rag 模型
    retriever = init_rag_retriever(rag_config=WEB_CONFIGS.RAG_CONFIG_PATH, db_path=WEB_CONFIGS.RAG_WORK_DIR)
    return retriever


def rebuild_rag_db(db_name="default"):

    # 重新生成 RAG 向量数据库
    gen_rag_db(force_gen=True)

    # 重新加载 retriever
    RAG_RETRIEVER.pop(db_name)
    RAG_RETRIEVER.get(fs_id=db_name, config_path=WEB_CONFIGS.RAG_CONFIG_PATH, work_dir=WEB_CONFIGS.RAG_WORK_DIR)


if WEB_CONFIGS.ENABLE_RAG:
    RAG_RETRIEVER = load_rag_model()
else:
    RAG_RETRIEVER = None