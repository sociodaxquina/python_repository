## periodic actions
#Link: https://stackoverflow.com/questions/8600161/executing-periodic-actions-in-python
#Comments: executes foo() every 10 seconds

import time, threading
import sys

def foo():
    print(time.ctime())
    threading.Timer(10, foo).start()

foo()

#output:
#   Wed Feb 28 11:10:15 2018
#   Wed Feb 28 11:10:25 2018
#   Wed Feb 28 11:10:35 2018
#   Wed Feb 28 11:10:45 2018
#   Wed Feb 28 11:10:55 2018
#   (...)
