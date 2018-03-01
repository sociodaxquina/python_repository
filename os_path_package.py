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


##Example: HTM 
import os
import pandas as pd
import numpy as np

#Identify /data directory
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.split(file_path)[0]
dataDir_path = root_path + '/data'
print(dataDir_path)

#full path of each file
for root, dirs, files in os.walk(dataDir_path):
    for name in files:
       name = str(name)
       name = os.path.realpath(os.path.join(root,name))
       print name

