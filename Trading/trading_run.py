##Trading
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import trading_parser, trading_utils, trading_finantialUtils, trading_ML

##Parameters
depvar1=0.75
depvar2=1 # ! not implemented yet
depvar3=1 # ! not implemented yet

value='no'

odd_min=1.20
odd_max=1000

leagues=np.array(['ING','ESP','PT'])

stk_per=0.20
banca_inv=600
banca_max=800 #1140
banca_min=300

plot_title='Season xx'
plot_var_1='Bets'
plot_var_2='Banca [E]'

##Import input data
dataDir_path = os.path.dirname(os.path.realpath(__file__))
filename = 'gameWeekSample.xlsx'
inputData_path = dataDir_path + '/' + filename

df_raw = pd.read_excel(inputData_path)
df = df_raw

numLines_df = df.shape[0]
#discard unused depvar columns -> They give NaN values
df = df.drop(['depvar1', 'depvar2', 'depvar3', 'depvar4', 'depvar5', 'depvar6', 'depvar7'], axis=1)
print('[1] Import input data [OK]')

##Parsing
df = trading_parser.parser_league(df)
print('[2] Parsing input data [OK]')

##model trainning
model1, model2, model3, model4, model5, model6, model7 = trading_ML.main_train()
print('[3] Train model [OK]')

##model prediction
y_score = trading_ML.main_predict(df, model4)
df['predicted: depvar1=1'] = y_score
#print(df.head(10))
print('[4] Input data prediction [OK]')

##Adding odds
#[!] requires work
df['odd_home'] = df_raw.loc[:,'Odd Home']
df['odd_home'] = df['odd_home'].fillna(0)

##Bet flags
flag_depvar1_col = trading_utils.b_flag_depvar1_col(
	df, 
	numLines_df, 
	depvar1
	)
df['b_flag_depvar1'] = flag_depvar1_col

##Betting simulation
stk_col, profit_col, banca_actual = trading_finantialUtils.simulation(
	df, 
	leagues, 
	odd_min, 
	odd_max, 
	stk_per, 
	banca_inv, 
	banca_max, 
	banca_min
	)
               
df['stk'] = stk_col
df['profit'] = profit_col
df['banca'] = banca_actual

##Simulation finantial statistics
banca_sim_stats, banca_sim_stats_headers = trading_finantialUtils.banca_sim_stats(df)

print(df.loc[:,[
	'var1_04', 
	'var1_05', 
	'var1_08', 
	'predicted: depvar1=1', 
	'odd_home', 
	'stk', 
	'profit',
	'banca']]
	)
print(banca_sim_stats)
