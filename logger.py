
import datetime, logging, sys, os, time
from functools import wraps
LOG_DIR = './logs'

def logger(log_name: str = ''):
    def _logger(func):
        @wraps(func)
        def inner_function(*args, **kwargs):
            print = get_logger(log_name).info
            ret = func(*args, **kwargs)
            return ret
        return inner_function
    return _logger

def get_logger(name: str = '', save_dir : str = LOG_DIR) :
    # 1. 저장할 폴더 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 로그 파일 이름 생성 (예: log_20231025_153000.txt)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"{name}_log_{timestamp}.txt")

    # 3. 로거 생성
    logger = logging.getLogger() # 루트 로거
    logger.setLevel(logging.INFO)
    
    # 중복 출력 방지 (이미 핸들러가 있으면 제거)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 포맷 설정 (시간 - 메시지)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 5. 핸들러 추가: 파일 출력 (FileHandler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. 핸들러 추가: 화면 출력 (StreamHandler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger