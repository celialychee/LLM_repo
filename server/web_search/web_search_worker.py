import shutil
from pathlib import Path

from loguru import logger

import torch
import yaml

from .web_search import WebSearch
from ..web_configs import WEB_CONFIGS

CONTEXT_MAX_LENGTH = 3000  # 上下文最大长度
# 判断是否需要回答
SCORING_QUESTION_TEMPLTE_CN = '“{}”\n请仔细阅读以上内容，判断句子是否表明需要查询某地实时或最新旅行攻略，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有目的并且是表示需要网络搜索或具有实时性要求得 10 分；缺少目的地直接得 0 分；缺少实时性要求直接得 0 分。直接提供得分不要解释。'
SCORING_QUESTION_TEMPLTE_EN = '"{}"\nPlease carefully read the above content and determine whether the sentence indicates a need to query real-time or latest travel strategies for a certain location. The result is represented by 0-10. Directly provide the score without explanation.'
# 判断材料关联程度
SCORING_RELAVANCE_TEMPLATE_CN = '问题：“{}”\n材料：“{}”\n请仔细阅读以上内容，判断问题和材料的关联度，用0～10表示。判断标准：非常相关得 10 分；完全没关联得 0 分。直接提供得分不要解释。\n'  # noqa E501
SCORING_RELAVANCE_TEMPLATE_EN = 'Question: "{}", Background Information: "{}"\nPlease read the content above carefully and assess the relevance between the question and the material on a scale of 0-10. The scoring standard is as follows: extremely relevant gets 10 points; completely irrelevant gets 0 points. Only provide the score, no explanation needed.'  # noqa E501
# 得出搜索关键词
KEYWORDS_TEMPLATE_CN = '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。搜索参数类型 string， 内容是短语或关键字，以空格分隔。\n你现在是一个旅行热点攻略查询助手，用户问“{}”，你打算通过谷歌搜索查询相关资料，请提供用于搜索的关键字或短语，不要解释直接给出关键字或短语。'  # noqa E501
KEYWORDS_TEMPLATE_EN = 'Google search is a general-purpose search engine that can be used to access the internet, look up encyclopedic knowledge, keep abreast of current affairs and more. Search parameters type: string, content consists of phrases or keywords separated by spaces.\nYou are now the assistant in the "{}" communication group. A user asked "{}", you plan to use Google search to find related information, please provide the keywords or phrases for the search, no explanation, just give the keywords or phrases.'  # noqa E501
# 得出最终结果
GENERATE_TEMPLATE_CN = '材料：“{}”\n 问题：“{}” \n 请仔细阅读参考材料回答问题。'  # noqa E501
GENERATE_TEMPLATE_EN = 'Background Information: "{}"\n Question: "{}"\n Please read the reference material carefully and answer the question.'  # noqa E501


class HotQuery():
    def __init(self, llm: ChatClient, language: str):
        self.context_max_length = CONTEXT_MAX_LENGTH
        if language == 'zh':
            self.SCORING_QUESTION_TEMPLTE = SCORING_QUESTION_TEMPLTE_CN
            self.SCORING_RELAVANCE_TEMPLATE = SCORING_RELAVANCE_TEMPLATE_CN
            self.KEYWORDS_TEMPLATE = KEYWORDS_TEMPLATE_CN
            self.GENERATE_TEMPLATE = GENERATE_TEMPLATE_CN
        else:
            self.SCORING_QUESTION_TEMPLTE = SCORING_QUESTION_TEMPLTE_EN
            self.SCORING_RELAVANCE_TEMPLATE = SCORING_RELAVANCE_TEMPLATE_EN
            self.KEYWORDS_TEMPLATE = KEYWORDS_TEMPLATE_EN
            self.GENERATE_TEMPLATE = GENERATE_TEMPLATE_EN
        self.max_length = self.context_max_length - 2 * len(
                self.GENERATE_TEMPLATE)

     def process(self, query: str):
        """Try web search."""
        
        if not WEB_CONFIGS.ENABLE_SEARCH:
            logger.info('disable web_search')
            return
        
        # 首先判断该问题是否应该结合搜索引擎回答
        prompt = self.SCORING_QUESTION_TEMPLTE.format(query)
        score = self.llm.generate_response(prompt)
        truth, logs = is_truth(llm=self.llm,
                                prompt=prompt,
                                throttle=5,
                                default=10,
                                backend='remote')
        # 判断不需要借助websearch回答
        if not truth:
            logger.info("no need use web search")
            return

        engine = WebSearch(config_path=self.config_path)

        prompt = self.KEYWORDS_TEMPLATE.format(query)
        # 生成搜索关键词
        search_keywords = self.llm.generate_response(prompt)
        results, error = engine.get(query=search_keywords, max_results=2)

        if error is not None:
            # sess.code = ErrorCode.WEB_SEARCH_FAIL
            logger.info("ErrorCode.WEB_SEARCH_FAIL")
            # yield sess
            return
        
        web_knowledge = ""
        refrences = []
        for result_id, result in enumerate(results):
            result.cut(0, self.max_length)
            prompt = self.SCORING_RELAVANCE_TEMPLATE.format(
                query, result.brief)
            # truth, logs = is_truth(llm=self.llm, prompt=prompt, throttle=5, default=10, backend='puyu')
            truth, logs = is_truth(llm=self.llm,
                                   prompt=prompt,
                                   throttle=5,
                                   default=10,
                                   backend='remote')
            # sess.debug['WebSearchNode_relavance_{}'.format(article_id)] = logs
            if truth:
                web_knowledge += result.content
                web_knowledge += '\n'
                references.append(result.source)

        web_knowledge = web_knowledge[0:self.max_length].strip()
        if len(web_knowledge) < 1:
            # sess.code = ErrorCode.NO_SEARCH_RESULT
            logger.info("ErrorCode.NO_SEARCH_RESULT")
            # yield sess
            return

        prompt = self.GENERATE_TEMPLATE.format(web_knowledge, query)
        # sess.response = self.llm.generate_response(prompt=prompt, history=sess.history, backend="puyu"
        # response = self.llm.generate_response(prompt=prompt,
        #                                            history=sess.history,
        #                                            backend='remote')
        # sess.code = ErrorCode.SUCCESS
        logger.info("ErrorCode.SUCCESS")
        # yield sess
        return prompt