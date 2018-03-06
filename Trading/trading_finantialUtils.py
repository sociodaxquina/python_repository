##trading_finantialUtils.py
import numpy as np


def simulation(DataFrame, leagues, odd_min, odd_max, stk_per, banca_inv, banca_max, banca_min):
	df = DataFrame
	banca_actual=[]
	profit_col = []
	stk_col=[]
	first_b_flag=0
	for i in range(0, df.shape[0]):
	    home_score = df.loc[i,'var1_04']
	    away_score = df.loc[i,'var1_05']
	    cond1 = (df.loc[i,'b_flag_depvar1'] == 1)
	    cond2 = (df.loc[i,'var1_08'] in leagues)
	    cond3 = (df.loc[i,'odd_home'] >= odd_min)
	    cond4 = (df.loc[i,'odd_home'] <= odd_max)

	    if (cond1 & cond2 & cond3 & cond4):

	         if first_b_flag == 0:
	              banca_temp_0 = banca_inv
	              stk = stk_per*banca_temp_0
	              if (home_score > away_score):
	                   profit_temp = stk * df.loc[i,'odd_home'] - stk                   
	              else:
	                   profit_temp = -stk
	              profit_temp, stk = banca_threshold(
	              	banca_temp_0, 
	              	banca_max, 
	              	banca_min, 
	              	profit_temp,
	              	stk
	              	)
	              banca_temp_1 = banca_temp_0 + profit_temp
	              stk_col = np.append(stk_col, round(stk, 2))
	              profit_col = np.append(profit_col, round(profit_temp, 2))
	              banca_actual = np.append(banca_actual, round(banca_temp_1, 2))
	              first_b_flag = 1

	         elif first_b_flag == 1:
	              banca_temp_0 = banca_actual[-1]
	              stk = stk_per * banca_actual[i-1]
	              if (home_score > away_score):
	                   profit_temp = stk * df.loc[i,'odd_home'] - stk 
	              else:
	                   profit_temp = -stk
	              profit_temp, stk = banca_threshold(
	              	banca_temp_0, 
	              	banca_max, 
	              	banca_min, 
	              	profit_temp,
	              	stk
	              	)
	              banca_temp_1 = banca_temp_0 + profit_temp
	              profit_col = np.append(profit_col, round(profit_temp, 2))
	              banca_actual = np.append(banca_actual, round(banca_temp_1, 2))
	              stk_col = np.append(stk_col, round(stk))
	    else:
	         stk=0
	         stk_col = np.append(stk_col, stk)
	         if first_b_flag == 0:
	              banca_temp_0 = banca_inv
	              profit_col = np.append(profit_col, 0)
	              banca_actual = np.append(banca_actual, banca_temp_0)
	              first_b_flag = 1
	         elif first_b_flag == 1:
	              banca_temp_0 = banca_actual[i-1]
	              profit_col = np.append(profit_col, 0)
	              banca_actual = np.append(banca_actual,banca_temp_0)

	return stk_col, profit_col, banca_actual

def banca_threshold(banca_temp, banca_max, banca_min, profit_temp, stk):
	cond1 = (banca_temp < banca_min)
	cond2 = (banca_temp > banca_max)
	if cond1 or cond2:
		profit_temp = 0
		stk = 0
	else:
		profit_temp = profit_temp
		stk = stk

	return profit_temp, stk

def banca_sim_stats(DataFrame):
	df = DataFrame

	banca_array = np.array(df['banca'])

	banca_sim_max = round(np.max(banca_array),2)
	banca_sim_min = round(np.min(banca_array),2)
	banca_sim_avg = round(np.average(banca_array),2)
	banca_sim_std = round(np.std(banca_array),2)
	banca_sim_endProfit = round(df.loc[(df.shape[0]-1),'banca'],2)
	banca_sim_numBets = df[df.loc[:,'stk'] > 0].shape[0]

	banca_sim_stats_headers = ['banca_sim_min', 'banca_sim_max', 'banca_sim_avg', 'banca_sim_std', 'banca_sim_endProfit', 'banca_sim_numBets']
	banca_sim_stats = [banca_sim_min, banca_sim_max, banca_sim_avg, banca_sim_std, banca_sim_endProfit, banca_sim_numBets]

	#print(banca_sim_stats_headers)
	#print(banca_sim_stats)

	return banca_sim_stats, banca_sim_stats_headers