import os
import logging
import logging.config

LogConfig = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'  # ,
            # 'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'filters': {},
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': None,  # written by system/engine/strategy
            'formatter': 'standard',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'console'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

class SelfLogger(object):
    def __init__(self,filename=None,name=None):
        if not filename:
            print('Not an avaliable log file name, exited!')
            exit(0)
        if not os.path.exists(filename):
            try:
                os.makedirs(os.path.split(filename)[0])
            except Exception as e:
                print(e)
                exit(0)
        LogConfig['handlers']['default']['filename'] = filename
        logging.config.dictConfig(LogConfig)
        if name:
            self.logger = logging.getLogger(name)
        else:
            self.logger = logging.getLogger(filename)

    def info(self,content):
        self.logger.info(content)

    def warning(self,content):
        self.logger.warning(content)

    def debug(self,content):
        self.logger.debug(content)

    def error(self,content):
        self.logger.error(content)

    def fatal(self,content):
        self.logger.fatal(content)