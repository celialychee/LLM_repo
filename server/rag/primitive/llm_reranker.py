# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import pdb
import requests
from typing import List, Tuple

import numpy as np

from .chunk import Chunk
from .embedder import Embedder
from .rpm import RPM

import sys
# sys.path.append("../../")
# from server import WEB_CONFIGS
sys.path.append(os.getcwd())
from server.web_configs import WEB_CONFIGS


class LLMReranker:
    _type: str
    topn: int

    def __init__(
            self,
            # model_config: dict,
            topn: int = 10):
        # 模型路径
        # model_name_or_path = model_config['reranker_model_path']
        model_name_or_path = os.path.join(WEB_CONFIGS.RAG_MODEL_DIR, WEB_CONFIGS.RAG_RERANKER_DIR)
        # model_name_or_path = os.path.join(WEB_CONFIGS.RAG_MODEL_DIR, model_config['reranker_model_path'])
        self._type = self.model_type(model_path=model_name_or_path)
        self.topn = topn

        if 'bge' in self._type:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16).eval().to('cuda')
        # BCE
        elif 'bce' in self._type:
            from BCEmbedding import RerankerModel
            self.bce_client = RerankerModel(
                model_name_or_path=model_name_or_path, use_fp16=True)
        elif 'siliconcloud' in self._type:
            # api_token = model_config['api_token'].strip()
            api_token = WEB_CONFIGS.RAG_SILICONCLOUD_API_KEY.strip()
            if len(api_token) < 1:
                api_token = os.getenv('SILICONCLOUD_TOKEN')
                if api_token is None or len(api_token) < 1:
                    raise ValueError('siliconclud remote reranker api token is None')
            if 'Bearer' not in api_token:
                api_token = 'Bearer ' + api_token
            # api_rpm = max(1, int(model_config['api_rpm']))
            api_rpm = max(1, int(WEB_CONFIGS.RAG_API_RPM))
            self.client = {
                'api_token': api_token,
                'api_rpm': RPM(api_rpm)
            }

        else:
            raise ValueError('Unknown type {}'.format(self._type))


    @classmethod
    def model_type(self, model_path):
        """Check reranker model is LLM reranker or not."""
        if model_path.startswith('https'):
            return 'siliconcloud'        

        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            if 'bge-reranker-v2-minicpm-layerwise' in config_path.lower():
                return 'bge'
            return 'bce'
        try:
            with open(config_path) as f:
                if 'bge-reranker-v2-minicpm-layerwise' in json.loads(
                        f.read())['_name_or_path']:
                    return 'bge'
        except Exception as e:
            logger.warning(e)
        return 'bce'

    def _get_inputs(self, pairs, prompt=None, max_length=1024):
        """Build input tokens with query and chunks."""
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = '\n'
        # 给定query A和passage B, 查看A是否和B相关
        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            # 在某些自然语言处理任务中，
            # 查询（query）通常比段落（passage）更短，
            # 但其重要性可能更高。通过限制查询的最大长度，
            # 可以确保查询文本不会占用太多的空间，
            # 从而使段落文本有更多的空间来表达更多的信息。
            query_inputs = self.tokenizer(f'A: {query}',
                                          return_tensors=None,
                                          add_special_tokens=False,
                                          max_length=max_length * 3 // 4,
                                          truncation=True)
            passage_inputs = self.tokenizer(f'B: {passage}',
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length,
                                            truncation=True)
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',  # 确保在组合后的序列超过最大长度时，只截断 passage 部分
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False)
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return self.tokenizer.pad(inputs,
                                  # 长度对齐
                                  padding=True,
                                  max_length=max_length + len(sep_inputs) +
                                  len(prompt_inputs),
                                  # 确保输入长度是对 8 的倍数。
                                  # 这是因为在某些硬件上（如 GPU），处理长度为 8 的倍数的数据可以提高性能
                                  pad_to_multiple_of=8,
                                  # 返回 PyTorch 张量形式
                                  return_tensors='pt')

    def _sort(self, texts: List[str], query: str):
        """Rerank input texts, return descending indexes, indexes[0] is the
        nearest chunk."""
        pairs = []
        # 将查询和备选chunk组成相似查询对
        for text in texts:
            pairs.append([query, text])

        if 'bge' in self._type:
            import torch
            with torch.no_grad():
                inputs = self._get_inputs(pairs).to(self.model.device)
                all_scores = self.model(**inputs,
                                        return_dict=True,
                                        cutoff_layers=[28])
                scores = [
                    scores[:, -1].view(-1, ).float()
                    for scores in all_scores[0]
                ]
                scores = scores[0].cpu().numpy()
        elif 'bce' in self._type:
            # 计算相似度
            scores_list = self.bce_client.compute_score(pairs)
            scores = np.array(scores_list)
        else:
            self.client['api_rpm'].wait(silent=True)
            
            url = "https://api.siliconflow.cn/v1/rerank"
            payload = {
                "model": "netease-youdao/bce-reranker-base_v1",
                "query": query,
                "documents": texts,
                "return_documents": False,
                "max_chunks_per_doc": 832,
                "overlap_tokens": 32
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": self.client['api_token']
            }
            response = requests.post(url, json=payload, headers=headers)
            json_obj = json.loads(response.text)
            results = json_obj['results']
            indexes_list = [round(item['index']) for item in results]
            indexes = np.array(indexes_list).astype(np.int32)
            return indexes[0:self.topn]

        # get descending order
        # argsort默认升序
        # [::-1]反序变为降序
        # 返回索引 获取topn个最相似chunk的索引
        return scores.argsort()[::-1][0:self.topn]

    def rerank(self, query: str, chunks: List[Chunk]):
        """Rerank faiss search results."""
        if not chunks:
            return []

        texts = []
        for chunk in chunks:
            # 原本的chunk是包装的一个类，要取出其中的具体内容进行检索
            texts.append(chunk.content_or_path)

        # During reranking, we just take image path as text
        indexes = self._sort(texts=texts, query=query)
        return [chunks[i] for i in indexes]
