import shutil
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Response
from sse_starlette import EventSourceResponse

from ..web_configs import WEB_CONFIGS
from .modules.rag.rag_worker import rebuild_rag_db
from .server_info import SERVER_PLUGINS_INFO
from .utils import ChatItem, UploadProductItem, SalesInfo, streamer_sales_process

app = FastAPI()
