## inspect package
#Link: --
#Comments: Check where the package/module is being imported from

import inspect
from m import p

print(inspect.getmodule(p))

#
