##trading_utilities
import numpy as np

def b_flag_depvar1(match_id, DataFrame, depvar):
	df = DataFrame
	depvar1 = depvar
	id = match_id
	id_pred_depvar1 = df.loc[df['id']==id]['pred_depvar1']
	cond1 = (id_pred_depvar1 >= depvar1)
	if (cond1.item()):
	  b_flag=1
	else:
	  b_flag=0

	return b_flag

def b_flag_depvar1_col(DataFrame, numLines_df, depvar):
	depvar1 = depvar
	df = DataFrame
	numLines_df = numLines_df
	flag_depvar1_col=[]
	for d in range(0, numLines_df):
		match_id = df.loc[d,'id']
		match_id_flag = b_flag_depvar1(match_id, df, depvar1)
		flag_depvar1_col = np.append(flag_depvar1_col, match_id_flag)

	return flag_depvar1_col


