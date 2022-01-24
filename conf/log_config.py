import os
import time
from loguru import logger

# 每小时生成一个
from conf.config import Config


def should_rotate_by_hour(message, file):
    filepath = os.path.abspath(file.name)
    creation = os.path.getctime(filepath)
    now = message.record["time"].timestamp()
    maxtime = 60 * 60  # 1 hour in seconds
    return now - creation > maxtime

# 每天生成一个
def should_rotate_by_day(message, file):
    filepath = os.path.abspath(file.name)
    creation_timestamp = os.path.getctime(filepath)
    now_timestamp = message.record["time"].timestamp()
    # 当前日志创建时间和当前时间不是同一天，则新创建
    create = time.strftime("%Y-%m-%d", time.localtime(creation_timestamp))
    now = time.strftime("%Y-%m-%d", time.localtime(now_timestamp))
    if now == create:
        return False
    else:
        return True


def config():
    # 日志异步写入
    file_path =  os.path.abspath(os.path.dirname(__file__))
    logs_dir = os.path.join(file_path, "../logs/")
    if Config.get('log_dir'):
        logs_dir = Config.get('log_dir')
    logs_path = logs_dir + "file-{time:YYYY-MM-DD}.log"
    logger.add(logs_path, rotation=should_rotate_by_day, enqueue=True)
