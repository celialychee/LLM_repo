# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os
import shutil

import fitz
import pandas as pd
import requests
import textract
from bs4 import BeautifulSoup
from loguru import logger


class FileName:
    """Record file original name, state and copied filepath with text
    format."""

    def __init__(self, root: str, filename: str, _type: str):
        self.root = root
        self.prefix = filename.replace('/', '_')
        self.basename = os.path.basename(filename)  # 文件名
        self.origin = os.path.join(root, filename)  # 原始文件路径
        self.copypath = self.origin  # 创建数据库过程目录
        self._type = _type  # 记录文件类型
        self.state = True  # 记录最后操作状态
        self.reason = ''  # 记录操作失败原因

    def __str__(self):
        return '{},{},{},{}\n'.format(self.basename, self.copypath, self.state,
                                      self.reason)


class FileOperation:
    """Encapsulate all file reading operations."""

    def __init__(self):
        # 各类型文件后缀
        self.image_suffix = ['.jpg', '.jpeg', '.png', '.bmp']
        self.md_suffix = '.md'
        self.text_suffix = ['.txt', '.text']
        self.excel_suffix = ['.xlsx', '.xls', '.csv']
        self.pdf_suffix = '.pdf'
        self.ppt_suffix = '.pptx'
        self.html_suffix = ['.html', '.htm', '.shtml', '.xhtml']
        self.word_suffix = ['.docx', '.doc']
        # self.code_suffix = ['.py', '.cpp', '.h']
        self.normal_suffix = [self.md_suffix
                              ] + self.text_suffix + self.excel_suffix + [
                                  self.pdf_suffix
                              ] + self.word_suffix + [self.ppt_suffix
                                                      ] + self.html_suffix

    def save_image(self, uri: str, outdir: str):
        """Save image URI to local dir.

        Return None if failed.
        """
        images_dir = os.path.join(outdir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        md5 = hashlib.md5()
        md5.update(uri.encode('utf8'))
        uuid = md5.hexdigest()[0:6]
        filename = uuid + uri[uri.rfind('.'):]
        image_path = os.path.join(images_dir, filename)

        logger.info('download {}'.format(uri))
        try:
            if uri.startswith('http'):
                resp = requests.get(uri, stream=True)
                if resp.status_code == 200:
                    with open(image_path, 'wb') as image_file:
                        for chunk in resp.iter_content(1024):
                            image_file.write(chunk)
            else:
                shutil.copy(uri, image_path)
        except Exception as e:
            logger.debug(e)
            return None, None
        return uuid, image_path

    def get_type(self, filepath: str):
        """Get filetype depends on URI suffix."""
        filepath = filepath.lower()
        if filepath.endswith(self.pdf_suffix):
            return 'pdf'

        if filepath.endswith(self.md_suffix):
            return 'md'

        if filepath.endswith(self.ppt_suffix):
            return 'ppt'

        for suffix in self.image_suffix:
            if filepath.endswith(suffix):
                return 'image'

        for suffix in self.text_suffix:
            if filepath.endswith(suffix):
                return 'text'

        for suffix in self.word_suffix:
            if filepath.endswith(suffix):
                return 'word'

        for suffix in self.excel_suffix:
            if filepath.endswith(suffix):
                return 'excel'

        for suffix in self.html_suffix:
            if filepath.endswith(suffix):
                return 'html'

        # for suffix in self.code_suffix:
        #     if filepath.endswith(suffix):
        #         return 'code'
        return None

    def md5(self, filepath: str):
        # 生成一个哈希编码的文件名
        hash_object = hashlib.sha256()
        with open(filepath, 'rb') as file:
            chunk_size = 8192
            while chunk := file.read(chunk_size):
                hash_object.update(chunk)

        return hash_object.hexdigest()[0:8]

    def summarize(self, files: list):
        # 总结文件处理情况
        success = 0
        skip = 0
        failed = 0

        for file in files:
            if file.state:
                success += 1
            elif file.reason == 'skip':
                skip += 1
            else:
                # logger.info('{} {}'.format(file.origin, file.reason))
                failed += 1

            # logger.info('{} {}'.format(file.reason, file.copypath))
        logger.info('累计{}文件，成功{}个，跳过{}个，异常{}个'.format(len(files), success,
                                                      skip, failed))

    def scan_dir(self, repo_dir: str):
        # 将原始文件目录下的文件存储为FileName对象再打包承一个列表返回
        files = []
        for root, _, filenames in os.walk(repo_dir):
            for filename in filenames:
                _type = self.get_type(filename)
                if _type is not None:
                    files.append(
                        FileName(root=root, filename=filename, _type=_type))
        return files

    def read_pdf(self, filepath: str):
        # load pdf and serialize table
        # PyMuPDF(也被称为fitz库)来获取PDF文献的标题通常涉及读取PDF的元数据
        text = ''
        with fitz.open(filepath) as pages:
            # 将每一页的内容添加进来
            for page in pages:
                # 提取文本内容，而不包括图片或其他非文本元素
                text += page.get_text()
                # 找到页面中的表格
                tables = page.find_tables()
                for table in tables:
                    # 生成表格名字
                    tablename = '_'.join(
                        filter(lambda x: x is not None and 'Col' not in x,
                               table.header.names))
                    pan = table.to_pandas()
                    # 取出空白列将表格转变为json格式
                    json_text = pan.dropna(axis=1).to_json(force_ascii=False)
                    text += tablename
                    text += '\n'
                    text += json_text
                    text += '\n'
        return text

    def read_excel(self, filepath: str):
        table = None
        if filepath.endswith('.csv'):
            table = pd.read_csv(filepath)
        else:
            table = pd.read_excel(filepath)
        if table is None:
            return ''
        # 去除空白列将表格转变为json格式
        json_text = table.dropna(axis=1).to_json(force_ascii=False)
        return json_text

    # 读取各种类型文件中的文字内容
    def read(self, filepath: str):
        # 获取文件类型
        file_type = self.get_type(filepath)

        text = ''
        
        # 文件不存在则返回
        if not os.path.exists(filepath):
            return text, None

        try:
            # 文本类型文件内容直接读取
            if file_type == 'md' or file_type == 'text':
                with open(filepath) as f:
                    text = f.read()

            # pdf类型文件单独读取
            elif file_type == 'pdf':
                text += self.read_pdf(filepath)

            # 表格类型文件单独读取
            elif file_type == 'excel':
                text += self.read_excel(filepath)
            
            # Textract是一个强大的Python库,用于从各种文件格式中提取文本
            # pdf采用fitz处理更精确
            elif file_type == 'word' or file_type == 'ppt':
                # https://stackoverflow.com/questions/36001482/read-doc-file-with-python
                # https://textract.readthedocs.io/en/latest/installation.html
                text = textract.process(filepath).decode('utf8')
                if file_type == 'ppt':
                    text = text.replace('\n', ' ')

            elif file_type == 'html':
                with open(filepath) as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text += soup.text

        except Exception as e:
            logger.error((filepath, str(e)))
            return '', e
        text = text.replace('\n\n', '\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('  ', ' ')
        text = text.replace('  ', ' ')
        text = text.replace('  ', ' ')
        return text, None


if __name__ == '__main__':

    def get_pdf_files(directory):
        pdf_files = []
        # 遍历目录
        for root, dirs, files in os.walk(directory):
            for file in files:
                # 检查文件扩展名是否为.pdf
                if file.lower().endswith('.pdf'):
                    # 将完整路径添加到列表中
                    pdf_files.append(os.path.abspath(os.path.join(root, file)))
        return pdf_files

    # 将你想要搜索的目录替换为下面的路径
    pdf_list = get_pdf_files(
        '/home/khj/huixiangdou-web-online-data/hxd-bad-file')

    # 打印所有找到的PDF文件的绝对路径

    opr = FileOperation()
    for pdf_path in pdf_list:
        text, error = opr.read(pdf_path)
        print('processing {}'.format(pdf_path))
        if error is not None:
            print('')

        else:
            if text is not None:
                print(len(text))
            else:
                print('')