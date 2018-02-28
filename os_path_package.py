## os.path package
#Link: https://docs.python.org/2/library/os.path.html
#Comments: __file__ has metadata of the file.py

import os

realpath = os.path.realpath(__file__)
print(realpath)
#output: /folder1/folder2/folder3/file.py

rootpath = os.path.dirname(os.path.realpath(__file__))
print(rootpath)
#output: /folder1/folder2/folder3/

basename = os.path.basename(__file__)
print(basename)
#output: file.py

create_path = os.path.join('folder1','folder2','file.json')
print(create_path)
#output: /folder1/folder2/folder3/file.json



