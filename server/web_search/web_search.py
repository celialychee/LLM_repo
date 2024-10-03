"""Web search utils."""
import argparse
import asyncio
import json
import os
import time
import types
import pytoml
import requests
from bs4 import BeautifulSoup as BS
# from duckduckgo_search import DDGS
from loguru import logger
from readability import Document

from .file_operation import FileOperation
from .helper import check_str_useful

class Result:

    def __init__(self, content: str = '', source='', brief=''):
        self.content = content
        self.source = source
        if len(brief) < 1:
            self.brief = content
        else:
            self.brief = brief

    def __str__(self):
        return self.content

    def __len__(self):
        return len(self.content)

    def cut(self, start_index, end_index):
        self.source = self.source[start_index:end_index]


class WebSearch:
    """This class provides functionality to perform web search operations.

    Attributes:
        config_path (str): Path to the configuration file.
        retry (int): Number of times to retry a request before giving up.

    Methods:
        load_key(): Retrieves API key from the config file.
        load_save_dir(): Gets the directory path for saving results.
        google(query: str, max_results:int): Performs Google search for the given query and returns top max_article results.  # noqa E501
        save_search_result(query:str, results: list): Saves the search result into a text file.  # noqa E501
        get(query: str, max_results=1): Searches with cache. If the query already exists in the cache, return the cached result.  # noqa E501
    """

    def __init__(self, config_path: str, retry: int = 1, language:str='zh') -> None:
        """Initializes the WebSearch object with the given config path and
        retry count."""

        self.search_config = None
        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)
            self.search_config = types.SimpleNamespace(**config['web_search'])

        self.retry = retry
        self.language = language

    def load_key():
        try:
            api_key = os.environ.get('SERPER_API_KEY', api_key)
            if api_key is None:
                raise ValueError(
                    'Please set Serper API key either in the environment '
                    'as SERPER_API_KEY or pass it as `api_key` parameter.')
            return api_key
            # return self.search_config.serper_x_api_key
        except Exception as e:
            return ''

    def fetch_url(self, query: str, target_link: str, brief: str = ''):
        if not target_link.startswith('http'):
            return None

        logger.info(f'extract: {target_link}')
        try:
            content = ''
            # 如果是文件需要直接下载
            if target_link.lower().endswith(
                    '.pdf') or target_link.lower().endswith('.docx'):
                # download file and parse
                logger.info(f'download and parse: {target_link}')
                response = requests.get(target_link,
                                        stream=True,
                                        allow_redirects=True)

                save_dir = self.search_config.save_dir
                basename = os.path.basename(target_link)
                save_path = os.path.join(save_dir, basename)

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_opr = FileOperation()
                content, error = file_opr.read(filepath=save_path)
                if error is not None:
                    return error
                return Result(content=content,
                               source=target_link,
                               brief=brief)
            # 如果不是文件则采用readability库读取网页内容
            response = requests.get(target_link, timeout=30)
           
            doc = Document(response.text)
            content_html = doc.summary()
            title = doc.short_title()
            soup = BS(content_html, 'html.parser')

            if len(soup.text) < 4 * len(query):
                return None
            content = '{} {}'.format(title, soup.text)
            content = content.replace('\n\n', '\n')
            content = content.replace('\n\n', '\n')
            content = content.replace('  ', ' ')
            
            # 检查文件内容字符是否为有效字符
            if not check_str_useful(content=content):
                return None
            # 有效再返回结果
            return Result(content=content, source=target_link, brief=brief)
        except Exception as e:
            logger.error('fetch_url {}'.format(str(e)))
        return None

    # def ddgs(self, query: str, max_article: int):
    #     """Run DDGS search based on query."""
    #     results = DDGS().text(query, max_results=20)
    #     filter_results = []

    #     for domain in self.search_config.domain_partial_order:
    #         for result in results:
    #             if domain in result['href']:
    #                 filter_results.append(result)
    #                 break

    #     logger.debug('filter results: {}'.format(filter_results))
    #     articles = []
    #     for result in filter_results:
    #         a = self.fetch_url(query=query,
    #                            target_link=result['href'],
    #                            brief=result['body'])
    #         if a is not None and len(a) > 0:
    #             articles.append(a)
    #         if len(articles) > max_article:
    #             break
    #     return articles

    def google(self, query: str, max_results: int):
        """Executes a google search based on the provided query.

        Parses the response and extracts the relevant URLs based on the
        priority defined in the configuration file. Performs a GET request on
        these URLs and extracts the title and content of the page. The content
        is cleaned and added to the articles list. Returns a list of results.
        """
        url = 'https://google.serper.dev/search'

        if 'zh' in self.language:
            lang = 'zh-cn'
        else:
            lang = 'en'
        payload = json.dumps({'q': f'{query}', 'hl': lang})
        headers = {
            'X-API-KEY': load_key(),
            'Content-Type': 'application/json'
        }
        response = requests.request('POST',
                                    url,
                                    headers=headers,
                                    data=payload,
                                    timeout=5)  # noqa E501
        jsonobj = json.loads(response.text)
        logger.debug(jsonobj)
        # 优先选用设置的域名内的搜索结果
        keys = self.search_config.domain_partial_order
        urls = {}
        normal_urls = []

        for organic in jsonobj['organic']:
            link = ''
            logger.debug(organic)

            if 'link' in organic:
                link = organic['link']
            else:
                link = organic['sitelinks'][0]['link']

            for key in keys:
                if key in link:
                    if key not in urls:
                        urls[key] = [link]
                    else:
                        urls[key].append(link)
                    break
                else:
                    normal_urls.append(link)

        logger.debug(f'gather urls: {urls}')

        links = []
        for key in keys:
            if key in urls:
                links += urls[key]

        target_links = links[0:max_results]

        logger.debug(f'target_links:{target_links}')

        results = []
        for target_link in target_links:
            # network with exponential backoff
            a = self.fetch_url(query=query, target_link=target_link)
            if a is not None:
                results.append(a)

        return results

    def save_search_result(self, query: str, results: list):
        """Writes the search results (articles) for the provided query into a
        text file.

        If the directory does not exist, it creates one. In case of an error,
        logs a warning message.
        """
        try:
            save_dir = self.search_config.save_dir
            if save_dir is None:
                return

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            filepath = os.path.join(save_dir,  )

            text = ''
            if len(results) > 0:
                texts = [str(a) for a in results]
                text = '\n\n'.join(texts)
            with open(filepath, 'w', encoding='utf8') as f:
                f.write(text)
        except Exception as e:
            logger.warning(f'error while saving search result {str(e)}')

    def logging_search_query(self, query: str):
        """Logging search query to txt file."""

        save_dir = self.search_config.save_dir
        if save_dir is None:
            return

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, 'search_query.txt')
        with open(filepath, 'a') as f:
            f.write(query)
            f.write('\n')

    def get(self, query: str, max_results=1):
        """Executes a google search with cache.

        If the query already exists in the cache, returns the cached result. If
        an exception occurs during the process, retries the request based on
        the retry count. Sleeps for a random time interval between retries.
        """
        query = query.strip()
        query = query[0:32]

        try:
            self.logging_search_query(query=query)

            results = []
            # engine = self.search_config.engine.lower()
            # if engine == 'ddgs':
            #     articles = self.ddgs(query=query, max_article=max_article)

            # elif engine == 'serper':
            results = self.google(query=query, max_results=max_results)

            self.save_search_result(query=query, results=results)

            return results, None
        except Exception as e:
            logger.error(('web_search exception', query, str(e)))
            return [], Exception('search fail, please check TOKEN')
        return [], None


def parse_args():
    """Parses command-line arguments for web search."""
    parser = argparse.ArgumentParser(description='Web search.')
    parser.add_argument('--keywords',
                        type=str,
                        help='Keywords for search and parse.')
    parser.add_argument(
        '--config_path',
        default='config.ini',
        help='Feature store configuration path. Default value is config.ini')
    args = parser.parse_args()
    return args


def fetch_web_content(target_link: str):
    """Fetches and parses the content of the target URL.

    Extracts the main content and title from the HTML of the page. Returns the
    title and content as a single string.
    """
    response = requests.get(target_link, timeout=60)

    doc = Document(response.text)
    content_html = doc.summary()
    title = doc.short_title()
    soup = BS(content_html, 'html.parser')
    ret = '{} {}'.format(title, soup.text)
    return ret


# if __name__ == '__main__':
#     # https://developer.aliyun.com/article/679591 failed
#     # print(fetch_web_content('https://www.volcengine.com/theme/4222537-R-7-1'))
#     parser = parse_args()
#     s = WebSearch(config_path=parser.config_path)
#     print(
#         s.fetch_url(
#             query='',
#             target_link=
#             'http://www.lswz.gov.cn/html/xhtml/ztcss/zt-jljstj/images/clgszpj.pdf'
#         ))
#     print(
#         s.fetch_url(query='',
#                     target_link='https://zhuanlan.zhihu.com/p/699164101'))
#     print(s.get('LMDeploy 修改日志级别'))
#     print(
#         fetch_web_content(
#             'https://mmdeploy.readthedocs.io/zh-cn/latest/get_started.html'))