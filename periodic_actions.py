
https://stackoverflow.com/questions/8600161/executing-periodic-actions-in-python

import time, threading
import sys

def foo():
    print(time.ctime())
    threading.Timer(5, foo).start()

foo()
