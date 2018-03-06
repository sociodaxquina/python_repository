##trading strategy

import numpy as np
import pandas as pd

##Parameters
#Fixed parameters
value='no'

odd_min = 1.20
odd_max = 1000

banca_inv = 600
banca_max = 800
banca_min = 300

#Variable parameters
depvar1 = 0.75
leagues = np.array(['ING','ESP','PT'])
leagues_comb = np.array([
	['ING'],
	['ESP'],
	['PT'],
	['ING', 'ESP'],
	['ING', 'PT'],
	['ESP', 'PT'],
	['ING','ESP','PT'], 
	])
stk_per = 0.20

df_strategy = pd.DataFrame({'stk_per':[], 'depvar1':[], 'leagues':[]})
df_strategy['leagues'] = df_strategy['leagues'].astype(object)

n = 0
stk_per_col = []
depvar1_col = []
leagues_col = []
for leagues_iter in leagues_comb:
	for depvar_iter in np.arange(0.0, 1.0+0.1, 0.1):
		for stk_per_iter in np.arange(0.0, 1.0+0.1, 0.1):
			leagues_col = -1
			stk_per_col = np.append(stk_per_col, stk_per_iter)
			depvar1_col = np.append(depvar1_col, depvar_iter)
df_strategy.loc[:,'stk_per'] = stk_per_col
df_strategy.loc[:,'depvar1'] = depvar1_col
n+=1

for leagues_iter in leagues_comb:
	for depvar_iter in np.arange(0.0, 1.0+0.1, 0.1):
		for stk_per_iter in np.arange(0.0, 1.0+0.1, 0.1):
			df_strategy.at[i, 'leagues'] = leagues_iter


#df_strategy.loc[:,'leagues'] = leagues_col
print(df_strategy)





