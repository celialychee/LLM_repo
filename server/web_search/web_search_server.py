import shutil
from pathlib import Path
from typing import Dict, List

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Response
from loguru import logger

from ..web_configs import WEB_CONFIGS
from .web_search_worker import HotQuery

from pydantic import BaseModel


app = FastAPI()


class ChatItem(BaseModel):
    user_id: str  # User 识别号，用于区分不用的用户调用
    request_id: str  # 请求 ID，用于生成 TTS & 数字人
    prompt: List[Dict[str, str]]  # 本次的 prompt

@app.post("/search_hot")
async def get_search_hot(chat_item: ChatItem):
    # 获取用户原始输入
    status = "fail"
    raw_prompt = chat_item.prompt[-1]["content"]
    logger.info(f"raw_prompt: {raw_prompt}")
    hot_query = HotQuery(llm, 'zh')
    web_prompt = hot_query.process(raw_prompt)
    
    # 如果判断为需要搜索的问题,就把搜索得出来的结果结合原问题一起作为输入
    if web_prompt:
        status = "success"
        chat_item.prompt[-1]["content"] = web_prompt
    logger.info(f"web_prompt: {web_prompt}")
    chat_prompt = chat_item.prompt[-1]["content"]
    logger.info(f"chat_item.prompt: {chat_prompt}")
    
    return {"user_id": chat_item.user_id, "request_id": chat_item.request_id, "status": status, "result": chat_item.prompt}


@app.get("/search_hot/check")
async def check_server():
    return {"message": "server enabled"}
