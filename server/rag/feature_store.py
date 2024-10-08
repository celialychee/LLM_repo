# Copyright (c) OpenMMLab. All rights reserved.
"""extract feature and search with user query."""
import argparse
import json
import os
import pdb
import re
import shutil
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

import pytoml
from loguru import logger
from torch.cuda import empty_cache
from tqdm import tqdm

from .primitive import (ChineseRecursiveTextSplitter, Chunk, Embedder, Faiss,
                         FileName, FileOperation,
                         RecursiveCharacterTextSplitter, nested_split_markdown)
# from .helper import histogram
# from .llm_server_hybrid import start_llm_server
from .retriever import CacheRetriever, Retriever

from server import WEB_CONFIGS


def read_and_save(file: FileName):
    if os.path.exists(file.copypath):
        # already exists, return
        logger.info('already exist, skip load')
        return
    file_opr = FileOperation()
    logger.info('reading {}, would save to {}'.format(file.origin,
                                                      file.copypath))
    content, error = file_opr.read(file.origin)
    if error is not None:
        logger.error('{} load error: {}'.format(file.origin, str(error)))
        return

    if content is None or len(content) < 1:
        logger.warning('{} empty, skip save'.format(file.origin))
        return

    with open(file.copypath, 'w') as f:
        f.write(content)


class FeatureStore:
    """Tokenize and extract features from the project's documents, for use in
    the reject pipeline and response pipeline."""

    def __init__(self,
                 embedder: Embedder,
                 config_path: str = 'config.ini',
                 language: str = 'zh',
                 chunk_size=900,
                 analyze_reject=False,
                 rejecter_naive_splitter=False) -> None:
                #  override=False) -> None:
        """Init with model device type and config."""
        self.config_path = config_path
        self.reject_throttle = -1
        self.language = language
        # self.override = override
        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)['feature_store']
            self.reject_throttle = config['reject_throttle']

        logger.debug('loading text2vec model..')
        self.embedder = embedder
        self.retriever = None
        self.chunk_size = chunk_size
        self.analyze_reject = analyze_reject

        if rejecter_naive_splitter:
            raise ValueError(
                'The `rejecter_naive_splitter` option deprecated, please `git checkout v20240722`'
            )

        if analyze_reject:
            raise ValueError(
                'The `analyze_reject` option deprecated, please `git checkout v20240722`'
            )

        logger.info('init dense retrieval database with chunk_size {}'.format(chunk_size))

        if language == 'zh':
            self.text_splitter = ChineseRecursiveTextSplitter(
                keep_separator=True,
                is_separator_regex=True,
                chunk_size=chunk_size,
                chunk_overlap=32)
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=32)

    def parse_markdown(self, file: FileName, metadata: Dict):
        length = 0
        text = file.basename + '\n'

        with open(file.copypath, encoding='utf8') as f:
            text += f.read()
        if len(text) <= 1:
            return [], length

        chunks = nested_split_markdown(file.origin,
                                       text=text,
                                       chunksize=self.chunk_size,
                                       metadata=metadata)
        for c in chunks:
            length += len(c.content_or_path)
        return chunks, length

    def build_dense(self, files: list, work_dir: str, markdown_as_txt: bool=False):
        """Extract the features required for the response pipeline based on the
        document."""
        feature_dir = os.path.join(work_dir, 'db_dense')
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        file_opr = FileOperation()
        chunks = []

        for i, file in enumerate(files):
            if not file.state:
                continue
            metadata = {'source': file.origin, 'read': file.copypath}

            # If you need higher rejection precision, set `markdown_as_txt` as True
            if not markdown_as_txt and file._type == 'md':
                md_chunks, md_length = self.parse_markdown(file=file,
                                                           metadata=metadata)
                chunks += md_chunks
                file.reason = str(md_length)

            else:
                # now read pdf/word/excel/ppt text
                text, error = file_opr.read(file.copypath)
                if error is not None:
                    file.state = False
                    file.reason = str(error)
                    continue
                file.reason = str(len(text))
                text = file.prefix + text
                chunks += self.text_splitter.create_chunks(
                    texts=[text], metadatas=[metadata])

        if not self.embedder.support_image:
            filtered_chunks = list(filter(lambda x: x.modal=='text', chunks))
        else:
            filtered_chunks = chunks
        if len(chunks) < 1:
            return

        # self.analyze(filtered_chunks)
        Faiss.save_local(folder_path=feature_dir, chunks=filtered_chunks, embedder=self.embedder)

    def preprocess(self, files: List, work_dir: str):
        """Preprocesses files in a given directory. Copies each file to
        'preprocess' with new name formed by joining all subdirectories with
        '_'.

        Args:
            files (list): original file list.
            work_dir (str): Working directory where preprocessed files will be stored.  # noqa E501

        Returns:
            str: Path to the directory where preprocessed markdown files are saved.

        Raises:
            Exception: Raise an exception if no markdown files are found in the provided repository directory.  # noqa E501
        """
        preproc_dir = os.path.join(work_dir, 'preprocess')
        if not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir)

        pool = Pool(processes=8)
        file_opr = FileOperation()
        for idx, file in enumerate(files):
            if not os.path.exists(file.origin):
                file.state = False
                file.reason = 'skip not exist'
                continue

            if file._type == 'image':
                file.state = False
                file.reason = 'skip image'

            elif file._type in ['pdf', 'word', 'excel', 'ppt', 'html']:
                # read pdf/word/excel file and save to text format
                md5 = file_opr.md5(file.origin)
                file.copypath = os.path.join(preproc_dir,
                                             '{}.text'.format(md5))
                pool.apply_async(read_and_save, (file, ))

            elif file._type in ['md', 'text']:
                # rename text files to new dir
                md5 = file_opr.md5(file.origin)
                file.copypath = os.path.join(
                    preproc_dir,
                    file.origin.replace('/', '_')[-84:])
                try:
                    shutil.copy(file.origin, file.copypath)
                    file.state = True
                    file.reason = 'preprocessed'
                except Exception as e:
                    file.state = False
                    file.reason = str(e)

            else:
                file.state = False
                file.reason = 'skip unknown format'
        pool.close()
        logger.debug('waiting for file preprocess finish..')
        pool.join()

        # check process result
        for file in files:
            if file._type in ['pdf', 'word', 'excel']:
                if os.path.exists(file.copypath):
                    file.state = True
                    file.reason = 'preprocessed'
                else:
                    file.state = False
                    file.reason = 'read error'

    def initialize(self, files: list, work_dir: str):
        """Initializes response and reject feature store.

        Only needs to be called once. Also calculates the optimal threshold
        based on provided good and bad question examples, and saves it in the
        configuration file.
        """
        logger.info(
            'initialize response and reject feature store, you only need call this once.'  # noqa E501
        )
        self.preprocess(files=files, work_dir=work_dir)
        # build dense retrieval refusal-to-answer and response database
        self.build_dense(files=files, work_dir=work_dir)

def test_reject(retriever: Retriever, sample: str = None):
    """Simple test reject pipeline."""
    if sample is None:
        real_questions = [
            "今天天气怎么样？",
            "今天吃什么？",
            "你真漂亮",
            "用科研仪器需要注意什么？",
            "啊，不是吧",
            "你能干什么"
        ]
    else:
        with open(sample) as f:
            real_questions = json.load(f)

    for example in real_questions:
        relative, score = retriever.is_relative(example)

        if relative:
            logger.warning(f'process query: {example}')
        else:
            logger.error(f'reject query: {example}')

        if sample is not None:
            if relative:
                with open('workdir/positive.txt', 'a+') as f:
                    f.write(example)
                    f.write('\n')
            else:
                with open('workdir/negative.txt', 'a+') as f:
                    f.write(example)
                    f.write('\n')
    empty_cache()


def test_query(retriever: Retriever, sample: str = None):
    """Simple test response pipeline."""
    from texttable import Texttable
    if sample is not None:
        with open(sample) as f:
            real_questions = json.load(f)
        logger.add('logs/feature_store_query.log', rotation='4MB')
    else:
        real_questions = [ 
            "圆明园背后有什么故事？",
            "黄鹤楼的历史背景是什么？",
            "你能说说八达岭长城背后的故事吗？",
            "今天天气怎么样？",
            "今天吃什么？",
            "你真漂亮",
            "用科研仪器需要注意什么？",
            "啊，不是吧",
            "你能干什么"
            ]

    table = Texttable()
    table.set_cols_valign(['t', 't', 't', 't'])
    table.header(['Query', 'State', 'Part of Chunks', 'References'])

    for example in real_questions:
        example = example[0:400]
        chunks, context, refs = retriever.query(example)
        if chunks:
            table.add_row(
                [example, 'Accepted', chunks[0:100] + '..', ','.join(refs)])
        else:
            table.add_row([example, 'Rejected', 'None', 'None'])
        empty_cache()

    logger.info('\n' + table.draw())
    empty_cache()

def gen_vector_db(config_path, source_dir, work_dir, test_mode=False, update_throttle=False):
    cache = CacheRetriever()
    fs_init = FeatureStore(embedder=cache.embedder,
                           config_path=config_path)
                        #    override=args.override)

    # walk all files in repo dir
    file_opr = FileOperation()
    files = file_opr.scan_dir(repo_dir=source_dir)
    fs_init.initialize(files=files, work_dir=work_dir)
    file_opr.summarize(files)
    del fs_init

    # update reject throttle
    if update_throttle:
        retriever = cache.get(config_path=config_path, work_dir=work_dir)
        # if retriever.kg.is_available():
        #     start_llm_server(args.config_path)

        with open(os.path.join('./server/rag/', 'resource', 'good_questions.json')) as f:
            good_questions = json.load(f)
        with open(os.path.join('./server/rag/', 'resource', 'bad_questions.json')) as f:
            bad_questions = json.load(f)
        retriever.update_throttle(config_path=config_path,
                                good_questions=good_questions,
                                bad_questions=bad_questions)

        cache.pop('default')

    # test
    if test_mode:
        retriever = cache.get(config_path=config_path, work_dir=work_dir)
        test_reject(retriever, args.sample)
        test_query(retriever, args.sample)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Feature store for processing directories.")
    parser.add_argument("--work_dir", type=str, default="./server/rag/work_dir", help="Working directory.")
    parser.add_argument("--repo_dir", type=str, default="./server/rag/repo_dir", help="Root directory where the repositories are located.")
    parser.add_argument(
        "--config_path", default="./server/rag/config.ini", help="Feature store configuration path. Default value is config.ini"
    )
    parser.add_argument(
        "--good_questions",
        default="./server/rag/resource/good_questions.json",
        help="Positive examples in the dataset. Default value is resource/good_questions.json",  # noqa E251  # noqa E501
    )
    parser.add_argument(
        "--bad_questions",
        default="./server/rag/resource/bad_questions.json",
        help="Negative examples json path. Default value is resource/bad_questions.json",  # noqa E251  # noqa E501
    )
    parser.add_argument(
        "--sample",
        help="Input an json file, save reject and search output.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    gen_vector_db(args.config_path, args.repo_dir, args.work_dir, test_mode=True, update_throttle=True)   
