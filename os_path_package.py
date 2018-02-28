## os.path package
#Comments:
#   __file__ has metadata of the file.py

import os

realpath = os.path.realpath(__file__)
print(realpath)
#output: /data/PEDRO/nupic_scripts/testes.py

rootpath = os.path.dirname(os.path.realpath(__file__))
print(rootpath)
#output: /data/PEDRO/nupic_scripts/testes.py



