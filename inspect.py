## inspect package
#Link: https://docs.python.org/2/library/os.path.html
#Comments: __file__ has metadata of the file.py

import os

realpath = os.path.realpath(__file__)
print(realpath)
#output: /folder1/folder2/folder3/file.py
