import pytest

import socket
from collections import deque

from multiprocessing import Process, Queue
import logging
from logging import handlers
from contextlib import contextmanager

STOP_SIGNAL = "Stop Read"

@pytest.fixture()
def logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture()
def caplog_workaround():
    
    
    @contextmanager
    def ctx():
        logger_queue = Queue()
        logger = logging.getLogger()
        logger.addHandler(handlers.QueueHandler(logger_queue))
        yield
        logger.removeHandler(handlers.QueueHandler(logger_queue))
        while not logger_queue.empty():
            log_record: logging.LogRecord = logger_queue.get()
            logger._log(
                level=log_record.levelno,
                msg=log_record.message,
                args=log_record.args,
                exc_info=log_record.exc_info,
            )
            if(log_record.message == STOP_SIGNAL):
                break
            

    return ctx