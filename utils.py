import telegram

def create_logger():
    import logging
    logger_file_name = r"log.txt"
    logging.basicConfig(
        format='%(asctime)s\t%(levelname)s\t(%(name)s)\t%(message)s',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(logger_file_name)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t(%(name)s)\t%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

telegram_token = ""  # put here your telegram token
chat_id = ""   # put here id of telegram channel
bot = telegram.Bot(token=telegram_token)

def send2telegramm(message):
    try:
        _ = bot.send_message(chat_id, message)
    except:
        #logger.warning("I could not sent the message to telegram")
        pass

