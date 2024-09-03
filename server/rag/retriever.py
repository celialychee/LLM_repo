# Copyright (c) OpenMMLab. All rights reserved.
"""extract feature and search with user query."""
import json
import os
import pdb
import time
from typing import Any, Union, Tuple, List

import numpy as np
import pytoml
from loguru import logger
from sklearn.metrics import precision_recall_curve

from .primitive import Embedder, Faiss, LLMReranker, Query, Chunk

from .primitive import FileOperation
# from .helper import QueryTracker
# from .kg import KnowledgeGraph

class Retriever:
    """Tokenize and extract features from the project's chunks, for use in the
    reject pipeline and response pipeline."""

    # def __init__(self, config_path: str, embedder: Any, reranker: Any,
    #              work_dir: str, reject_throttle: float) -> None:
    def __init__(self, embedder: Any, reranker: Any,
                 work_dir: str, reject_throttle: float) -> None:
        """Init with model device type and config."""
        # self.config_path = config_path
        self.reject_throttle = reject_throttle

        self.embedder = embedder
        self.reranker = reranker
        self.faiss = None

        if not os.path.exists(work_dir):
            logger.warning('!!!warning, workdir not exist.!!!')
            return

        # load prebuilt knowledge graph gpickle
        # self.kg = KnowledgeGraph(config_path=config_path)

        # dense retrieval, load refusal-to-answer and response feature database
        dense_path = os.path.join(work_dir, 'db_dense')
        print("dense_path", dense_path)
        if not os.path.exists(dense_path):
            logger.warning('retriever is None, skip load faiss')
            self.faiss = None
        else:
            # 加载向量数据库
            self.faiss = Faiss.load_local(dense_path)

    def update_throttle(self,
                        config_path: str = 'config.ini',
                        good_questions=[],
                        bad_questions=[]):
        """Update reject throttle based on positive and negative examples."""

        if len(good_questions) == 0 or len(bad_questions) == 0:
            raise Exception('good and bad question examples cat not be empty.')
        questions = good_questions + bad_questions
        predictions = []
        self.reject_throttle = -1

        for question in questions:
            # score是相关度最高的文本的相似度分数，没有大于阈值的文本则为-1
            # _, score = self.is_relative(query=question,
            #                            enable_kg=True, enable_threshold=False)
            _, score = self.is_relative(query=question, enable_threshold=False)
            predictions.append(max(0, score))

        labels = [1 for _ in range(len(good_questions))
                  ] + [0 for _ in range(len(bad_questions))]
        precision, recall, thresholds = precision_recall_curve(
            labels, predictions)

        # get the best index for sum(precision, recall)
        # 不同阈值得到的精准度和召回度求和，找到和最大情况下的阈值
        sum_precision_recall = precision[:-1] + recall[:-1]
        index_max = np.argmax(sum_precision_recall)
        optimal_threshold = max(thresholds[index_max], 0.0)

        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)
        # 更新阈值 相关度大于这个阈值才会回答
        config['feature_store']['reject_throttle'] = float(optimal_threshold)
        with open(config_path, 'w', encoding='utf8') as f:
            pytoml.dump(config, f)
        logger.info(
            f'The optimal threshold is: {optimal_threshold}, saved it to {config_path}'  # noqa E501
        )

    def text2vec_retrieve(self, query: Union[Query, str]):
        """Retrieve chunks by text2vec model or knowledge graph. 
        
        Args:
            query (Query): The multimodal question asked by the user.
        
        Returns:
            List[Chunk]: ref chunks.
        """
        if type(query) is str:
            query = Query(text=query)

        # graph_delta = 0.0
        # if self.kg.is_available():
        #     try:
        #         docs = self.kg.retrieve(query=query.text)
        #         graph_delta = 0.2 * min(100, len(docs)) / 100
        #     except Exception as e:
        #         logger.warning(str(e))
        #         logger.info('KG folder exists, but search failed, skip.')

        # threshold = self.reject_throttle - graph_delta
        threshold = self.reject_throttle
        pairs = self.faiss.similarity_search_with_query(self.embedder,
                                                        query=query, threshold=threshold)
        chunks = [pair[0] for pair in pairs]
        # 仅返回相关文本块信息列表
        return chunks

    def rerank_fuse(self, query: Union[Query, str], chunks: List[Chunk], context_max_length:int):
        """Rerank chunks and extract content
        
        Args:
            chunks (List[Chunk]): filtered chunks.
        
        Returns:
            str: Joined chunks, or empty string
            str: Matched context from origin file content
            List[str]: References
        """
        if type(query) is str:
            query = Query(text=query)

        rerank_chunks = self.reranker.rerank(query=query.text,
                                      chunks=chunks)

        file_opr = FileOperation()
        # Add file text to context, until exceed `context_max_length`
        # If `file_length` > `context_max_length` (for example file_length=300 and context_max_length=100)
        # then centered on the chunk, read a length of 200
        splits = []
        context = ''
        references = []
        for idx, chunk in enumerate(rerank_chunks):

            content = chunk.content_or_path
            splits.append(content)

            source = chunk.metadata['source']
            if '://' in source:
                # url
                file_text = content
            else:
                # file_text是chunk所在文件的内容
                file_text, error = file_opr.read(chunk.metadata['read'])
                if error is not None:
                    # read file failed, skip
                    continue

            logger.info('target {} content length {}'.format(
                source, len(file_text)))
            # 如果再加上这段文字context长度就超了
            if len(file_text) + len(context) > context_max_length:
                if source in references:
                    continue
                # 如果是新的参考文献就加入到参考文献列表
                references.append(source)

                # add and break
                # 计算还能加入的最大长度
                add_len = context_max_length - len(context)
                # 如果小于等于0则不能再加入
                if add_len <= 0:
                    break
                # 找到相关文本在文件中的起始位置
                content_index = file_text.find(content)
                if content_index == -1:
                    # content not in file_text
                    context += content
                    context += '\n'
                    context += file_text[0:add_len - len(content) - 1]
                else:
                    # 将相关文本前面相关文本长度的内容作为上下文
                    start_index = max(0,
                                      content_index - (add_len - len(content)))
                    context += file_text[start_index:start_index + add_len]
                # 到下一个块
                break
            # 如果没有参考过的文件且长度没有超过上下文文本限制，直接加入上下文
            if source not in references:
                context += file_text
                context += '\n'
                references.append(source)

        context = context[0:context_max_length]  # 保证上下文没有超过限制
        logger.debug('query:{} files:{}'.format(query, references))
        # 返回所有相似文本、上下文、参考文献
        return '\n'.join(splits), context, [
            os.path.basename(r) for r in references
        ]

    def query(self,
              query: Union[Query, str],
              context_max_length: int = 40000):
            #   tracker: QueryTracker = None):
        """Processes a query and returns the best match from the vector store
        database. If the question is rejected, returns None.

        Args:
            query (Query): The multimodal question asked by the user.
            context_max_length (int): Max contenxt length for LLM.
            tracker (QueryTracker): Log tracker.

        Returns:
            str: Matched chunks, or empty string
            str: Matched context from origin file content
            List[str]: References 
        """
        if type(query) is str:
            query = Query(text=query)

        if query.text is None or len(query.text) < 1 or self.faiss is None:
            return None, None, []

        if len(query.text) > 512:
            logger.warning('input too long, truncate to 512')
            query.text = query.text[0:512]

        high_score_chunks = self.text2vec_retrieve(query=query)
        # if tracker is not None:
        #     tracker.log('retrieve', [c.metadata['source'] for c in high_score_chunks])
        # 返回所有相似文本、上下文、参考文献
        return self.rerank_fuse(query=query, chunks=high_score_chunks, context_max_length=context_max_length)

    def is_relative(self,
                    query,
                    k=30,
                    # enable_kg=True,
                    enable_threshold=True) -> Tuple[bool, float]:
        """Is input query relative with knowledge base. Return true or false, and the maxisum score"""
        if type(query) is str:
            query = Query(text=query)

        if query.text is None or len(query.text) < 1 or self.faiss is None:
            raise ValueError('input query {}, faiss {}'.format(query, self.faiss))

        # graph_delta = 0.0
        # if not enable_kg and self.kg.is_available():
        #     try:
        #         docs = self.kg.retrieve(query=query.text)
        #         graph_delta = 0.2 * min(100, len(docs)) / 100
        #     except Exception as e:
        #         logger.warning(str(e))
        #         logger.info('KG folder exists, but search failed, skip.')

        # threshold = self.reject_throttle - graph_delta
        threshold = self.reject_throttle

        if enable_threshold:
            pairs = self.faiss.similarity_search_with_query(self.embedder, query=query, threshold=threshold)
        else:
            pairs = self.faiss.similarity_search_with_query(self.embedder, query=query, threshold=-1)
        if len(pairs) > 0:
            # 如果有大于阈值的相似度文本，就返回true以及相关度最高的文本的相似度分数
            return True, pairs[0][1]
        return False, -1

class CacheRetriever:

    def __init__(self,
                 cache_size: int = 4,
                 rerank_topn: int = 4):
        self.cache = dict()
        self.cache_size = cache_size

        # load text2vec and rerank model
        logger.info('loading test2vec and rerank models')
        self.embedder = Embedder()
        self.reranker = LLMReranker(topn=rerank_topn)

    def get(self,
            fs_id: str = 'default',
            config_path='config.ini',
            work_dir='work_dir'):
        """Get database by id."""

        if fs_id in self.cache:
            self.cache[fs_id]['time'] = time.time()
            return self.cache[fs_id]['retriever']

        with open(config_path, encoding='utf8') as f:
            reject_throttle = pytoml.load(
                f)['feature_store']['reject_throttle']

        if len(self.cache) >= self.cache_size:
            # drop the oldest one
            del_key = None
            min_time = time.time()
            for key, value in self.cache.items():
                cur_time = value['time']
                if cur_time < min_time:
                    min_time = cur_time
                    del_key = key

            if del_key is not None:
                del_value = self.cache[del_key]
                self.cache.pop(del_key)
                del del_value['retriever']

        # retriever = Retriever(config_path=config_path,
        retriever = Retriever(embedder=self.embedder,
                              reranker=self.reranker,
                              work_dir=work_dir,
                              reject_throttle=reject_throttle)
        self.cache[fs_id] = {'retriever': retriever, 'time': time.time()}
        return retriever

    def pop(self, fs_id: str):
        """Drop database by id."""

        if fs_id not in self.cache:
            return
        del_value = self.cache[fs_id]
        self.cache.pop(fs_id)
        # manually free memory
        del del_value
