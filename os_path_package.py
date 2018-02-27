## os.path package

import os

#__file__ has metadata of the file.py

realpath = os.path.realpath(__file__)
print(realpath)
'''
  > /data/PEDRO/nupic_scripts/testes.py
'''

rootpath = os.path.dirname(os.path.realpath(__file__))
print(rootpath)
'''
  > /data/PEDRO/nupic_scripts/testes.py
'''


