##Trading
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import trading_parser, trading_utils, trading_finantialUtils

##Variables control
depvar1=0.75
depvar2=1
depvar3=1

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
df=df.iloc[:,[0,2,3,4,5,6,7,8]]
df=df.iloc[0:numLines_df,:]

##Parsing
df = trading_parser.parser_league(df)

##model trainning
#[!] requires work

##Model prediction
#[!] requires work
df['pred_depvar1'] = df_raw.loc[:,'predicted: depvar1=1']

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

print(df.loc[:,[
	'var1_04', 
	'var1_05', 
	'var1_08', 
	'pred_depvar1', 
	'odd_home', 
	'stk', 
	'profit',
	'banca']]
	)
