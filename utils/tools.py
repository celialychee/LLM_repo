import os
import time
from pathlib import Path

def delete_old_files(directory, limit_time_s=60 * 60 * 2):
    """
    删除指定目录下超过一定时间的文件。

    :param directory: 要检查和删除文件的目录路径
    """

    if not Path(directory).exists():
        return

    # 获取当前时间戳
    current_time = time.time()

    # 遍历目录下的所有文件和子目录
    for file_path in Path(directory).iterdir():

        # 获取文件的修改时间戳
        file_mtime = os.path.getmtime(file_path)

        # 计算文件的年龄（以秒为单位）
        file_age_seconds = current_time - file_mtime

        # 检查文件是否超过 n 秒
        if file_age_seconds > limit_time_s:
            try:

                if file_path.is_dir():
                    shutil.rmtree(file_path)
                    continue

                # 删除文件
                file_path.unlink()
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")