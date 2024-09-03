import shutil
from pathlib import Path
from typing import Dict, List

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Response
from loguru import logger
# from sse_starlette import EventSourceResponse

from ..web_configs import WEB_CONFIGS
from .rag_worker import RAG_RETRIEVER, build_rag_prompt
# from .modules.rag.rag_worker import rebuild_rag_db
# from .server_info import SERVER_PLUGINS_INFO
# from .utils import ChatItem, UploadProductItem, SalesInfo, streamer_sales_process

from pydantic import BaseModel


app = FastAPI()


class ChatItem(BaseModel):
    user_id: str  # User 识别号，用于区分不用的用户调用
    request_id: str  # 请求 ID，用于生成 TTS & 数字人
    prompt: List[Dict[str, str]]  # 本次的 prompt

@app.post("/rag")
async def get_rag(chat_item: ChatItem):
    # 获取用户原始输入
    status = "fail"
    raw_prompt = chat_item.prompt[-1]["content"]
    logger.info(f"raw_prompt: {raw_prompt}")
    rag_prompt = build_rag_prompt(RAG_RETRIEVER, raw_prompt)
    
    # 如果RAG回答了该问题则结合RAG检索到的上下文和用户原始提问结合到一起
    if rag_prompt != "":
        status = "success"
        chat_item.prompt[-1]["content"] = rag_prompt
    logger.info(f"rag_prompt: {rag_prompt}")
    chat_prompt = chat_item.prompt[-1]["content"]
    logger.info(f"chat_item.prompt: {chat_prompt}")
    
    return {"user_id": chat_item.user_id, "request_id": chat_item.request_id, "status": status, "result": chat_item.prompt}


@app.get("/rag/check")
async def check_server():
    return {"message": "server enabled"}
