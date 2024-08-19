import json
from pathlib import Path
from time import sleep
import uuid

import requests
import streamlit as st
from server.web_configs import WEB_CONFIGS, API_CONFIG


def get_asr_api(wav_path, user_id="123"):
    # 获取 ASR 结果
    req_data = {
        "user_id": user_id,
        "request_id": str(uuid.uuid1()),
        "wav_path": wav_path,
    }

    print(req_data)

    res = requests.post(API_CONFIG.ASR_URL, json=req_data).json()
    return res["result"]