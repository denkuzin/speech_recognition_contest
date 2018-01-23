import telegram
import logging


LOG_FILE_NAME = "log.txt"
TELEGRAM_TOKEN = ""  # put here your telegram token
CHAT_ID = ""   # put here id of telegram channel


def create_logger():
    logging.basicConfig(
        format='%(asctime)s\t%(levelname)s\t(%(name)s)\t%(message)s',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(LOG_FILE_NAME)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t(%(name)s)\t%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def send2telegramm(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    try:
        _ = bot.send_message(CHAT_ID, message)
    except:
        #logger.warning("I could not sent the message to telegram")
        pass

