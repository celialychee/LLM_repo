import os
from dataclasses import dataclass


@dataclass
class WebConfigs:
    """
    项目所有的配置
    """

    # ==================================================================
    #                             LLM 模型配置
    # ==================================================================
    SALES_NAME: str = "乐乐喵"  # 启动的角色名
    # LLM_MODEL_DIR: str = r"./weights/llm_weights/"

    # ==================================================================
    #                               组件配置
    # ==================================================================
    # ENABLE_RAG: bool = True  # True 启用 RAG 检索增强，False 不启用
    # ENABLE_TTS: bool = True  # True 启动 tts，False 不启用
    # ENABLE_DIGITAL_HUMAN: bool = True  # True 启动 数字人，False 不启用
    # ENABLE_AGENT: bool = os.environ.get("ENABLE_AGENT", "true") == "true"  # True 启动 Agent，False 不启用
    ENABLE_ASR: bool = os.environ.get("ENABLE_ASR", "true") == "true"  # True 启动 语音转文字，False 不启用

    # DISABLE_UPLOAD: bool = os.getenv("DISABLE_UPLOAD") == "true"

    # ==================================================================
    #                               页面配置
    # ==================================================================
    # PRODUCT_IMAGE_HEIGHT: int = 400  # 商品图片高度
    # EACH_CARD_OFFSET: int = 100  # 每个商品卡片比图片高度多出的距离
    # EACH_ROW_COL: int = 2  # 商品页显示多少列

    # 定义用户和机器人头像路径
    # USER_AVATOR: str = "./assets/user.png"
    # ROBOT_AVATOR: str = "./assets/logo.png"

    # ==================================================================
    #                               商品配置
    # ==================================================================
    # PRODUCT_INSTRUCTION_DIR: str = r"./product_info/instructions"
    # PRODUCT_IMAGES_DIR: str = r"./product_info/images"

    # PRODUCT_INFO_YAML_PATH: str = r"./product_info/product_info.yaml"
    # PRODUCT_INFO_YAML_BACKUP_PATH: str = PRODUCT_INFO_YAML_PATH + ".bk"

    # ==================================================================
    #                             配置文件路径
    # ==================================================================
    # CONVERSATION_CFG_YAML_PATH: str = r"./configs/conversation_cfg.yaml"

    # ==================================================================
    #                               RAG 配置
    # ==================================================================
    # RAG_CONFIG_PATH: str = r"./configs/rag_config.yaml"
    # RAG_VECTOR_DB_DIR: str = r"./work_dirs/instruction_db"
    # PRODUCT_INSTRUCTION_DIR_GEN_DB_TMP: str = r"./work_dirs/instructions_gen_db_tmp"
    # RAG_MODEL_DIR: str = r"./weights/rag_weights/"

    # ==================================================================
    #                               TTS 配置
    # ==================================================================
    # TTS_WAV_GEN_PATH: str = r"./work_dirs/tts_wavs"
    # TTS_MODEL_DIR: str = r"./weights/gpt_sovits_weights/"
    # TTS_INF_NAME: str = "激动说话-列车巡游银河，我不一定都能帮上忙，但只要是花钱能解决的事，尽管和我说吧。.wav"

    # ==================================================================
    #                             数字人 配置
    # ==================================================================
    # DIGITAL_HUMAN_GEN_PATH: str = r"./work_dirs/digital_human"
    # DIGITAL_HUMAN_VIDEO_OUTPUT_PATH: str = r"./work_dirs/digital_human/vid_output"
    # DIGITAL_HUMAN_MODEL_DIR: str = r"./weights/digital_human_weights/"
    # DIGITAL_HUMAN_BBOX_SHIFT: int = 0
    # DIGITAL_HUMAN_VIDEO_PATH: str = r"./assets/lelemiao_digital_human_video.mp4"
    # DIGITAL_HUMAN_FPS: str = 25

    # ==================================================================
    #                             Agent 配置
    # ==================================================================
    # AGENT_WEATHER_API_KEY: str | None = os.environ.get("WEATHER_API_KEY", None)  # 天气 API Key
    # AGENT_DELIVERY_TIME_API_KEY: str | None = os.environ.get("DELIVERY_TIME_API_KEY", None)  # 快递查询 API Key

    # ==================================================================
    #                              ASR 配置
    # ==================================================================
    ASR_WAV_SAVE_PATH: str = r"./work_dirs/asr_wavs"
    ASR_MODEL_DIR: str = r"./models/asr/"


@dataclass
class ApiConfig:
    # ==================================================================
    #                               URL 配置
    # ==================================================================
    # USING_DOCKER_COMPOSE: bool = os.environ.get("USING_DOCKER_COMPOSE", "false") == "true"
    
    # 路由名字和 compose.yaml 服务名对应
    # TTS_ROUTER_NAME: str = "tts" if USING_DOCKER_COMPOSE else "0.0.0.0"
    # DIGITAL_ROUTER_NAME: str = "digital_human" if USING_DOCKER_COMPOSE else "0.0.0.0"
    # ASR_ROUTER_NAME: str = "asr" if USING_DOCKER_COMPOSE else "0.0.0.0"
    # LLM_ROUTER_NAME: str = "llm" if USING_DOCKER_COMPOSE else "0.0.0.0"
    # BASE_ROUTER_NAME: str = "base" if USING_DOCKER_COMPOSE else "0.0.0.0"
    ASR_ROUTER_NAME: str = "0.0.0.0"
    LLM_ROUTER_NAME: str = "127.0.0.1"
    
    # TTS_URL: str = f"http://{TTS_ROUTER_NAME}:8001/tts"
    # DIGITAL_HUMAN_URL: str = f"http://{DIGITAL_ROUTER_NAME}:8002/digital_human"
    ASR_URL: str = f"http://{ASR_ROUTER_NAME}:8003/asr"
    LLM_URL: str = f"http://{LLM_ROUTER_NAME}:23333"
    
    # CHAT_URL: str = f"http://{BASE_ROUTER_NAME}:8000/streamer-sales/chat"
    # UPLOAD_PRODUCT_URL: str = f"http://{BASE_ROUTER_NAME}:8000/streamer-sales/upload_product"
    # GET_PRODUCT_INFO_URL: str = f"http://{BASE_ROUTER_NAME}:8000/streamer-sales/get_product_info"
    # GET_SALES_INFO_URL: str = f"http://{BASE_ROUTER_NAME}:8000/streamer-sales/get_sales_info"
    # PLUGINS_INFO_URL: str = f"http://{BASE_ROUTER_NAME}:8000/streamer-sales/plugins_info"


# 实例化
WEB_CONFIGS = WebConfigs()
API_CONFIG = ApiConfig()