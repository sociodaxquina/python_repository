## Bet Simulator

##Libraries*********************************************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Variable Board****************************************************************
depvar1=0.75
depvar2=1
depvar3=1

value='no'

odd_min=1.20
odd_max=1000

leagues=np.array(['ING','ESP','PT'])

stk_per=0.20
banca_inv=600
banca_max=1140
banca_min=300

plot_title='Season xx'
plot_var_1='Bets'
plot_var_2='Banca [€]'

##train dataset*****************************************************************
path="C://Users//pfpereira//Downloads//dataset.xlsx"
df_raw=pd.read_excel(path)
df=df_raw.iloc[:,[0,2,3,4,5,6,7,8]]
league_dic={'Premier League':'ING','Primeira Liga':'PT','Primera DivisiÃ³n':'ESP', 'Primera DivisiÌ_n':'ESP'}
df=df.replace({"var1_08": league_dic})
#df=pd.DataFrame({'id':[2501874,2501869,2501866,2501829]})
#df['var1_04']=np.array([0, 1, 1,1])
#df['var1_05']=np.array([0, 0, 0, 0])
df=df.iloc[0:179,:]


##!!!!model trainning***************************************************************
df_raw.columns



##!!!model prediction**************************************************************
df['pred_depvar1']=df_raw.loc[:,'Predicted: depvar1=1']

##!!!adding odds*******************************************************************
df['odd_home']=df_raw.loc[:,'Odd Home']
df['odd_home']=df['odd_home'].fillna(0)


##functions and utilities*******************************************************
def b_flag_depvar1(match_id):
     id=match_id
     id_pred_depvar1=df.loc[df['id']==id]['pred_depvar1']
     cond1=(id_pred_depvar1>=depvar1)
     if (cond1.item()):
          b_flag=1
     else:
          b_flag=0
     return b_flag

flag_depvar1_col=[]
for d in range(0,df.shape[0]):
     match_id=df.loc[d,'id']
     flag_depvar1_col=np.append(flag_depvar1_col,b_flag_depvar1(match_id))

df['b_flag_depvar1']=flag_depvar1_col

#action***********************************************************************
banca_actual=[]
stk_col=[]
stk=-1
first_b_flag=0
for i in range(0,df.shape[0]):
     home_score=df.loc[i,'var1_04']
     away_score=df.loc[i,'var1_05']
     cond1=(df.loc[i,'b_flag_depvar1']==1)
     cond2=(df.loc[i,'var1_08'] in leagues)
     cond3=(df.loc[i,'odd_home']>=odd_min)
     cond4=(df.loc[i,'odd_home']<=odd_max)
     if (cond1 & cond2 & cond3 & cond4):
          if first_b_flag==0:
               banca_temp_0=banca_inv
               stk=stk_per*banca_temp_0
               stk_col=np.append(stk_col,stk)
               if (home_score>away_score):
                    profit_temp=stk*df.loc[i,'odd_home']-stk                   
               else:
                    profit_temp=-stk
               banca_temp_1=banca_temp_0+profit_temp
               banca_actual=np.append(banca_actual,banca_temp_1)
               first_b_flag=1
          elif first_b_flag==1:
               banca_temp_0=banca_actual[-1]
               stk=stk_per*banca_actual[i-1]
               if (home_score>away_score):
                    profit_temp=stk*df.loc[i,'odd_home']-stk 
               else:
                    profit_temp=-stk
               banca_temp_1=banca_temp_0+profit_temp
               banca_actual=np.append(banca_actual,banca_temp_1)
               stk_col=np.append(stk_col,stk)
     else:
          stk=0
          stk_col=np.append(stk_col,stk)
          if first_b_flag==0:
               banca_temp_0=banca_inv
               banca_actual=np.append(banca_actual,banca_temp_0)
               first_b_flag=1
          elif first_b_flag==1:
               banca_temp_0=banca_actual[i-1]
               banca_actual=np.append(banca_actual,banca_temp_0)
               
df['stk']=stk_col
df['banca']=banca_actual

print(df)

#plot**************************************************************************
'''
tick_min=0
tick_max=df.shape[0]
plt.plot(df['banca'])
plt.style.use("ggplot")      
plt.xlabel('{}'.format(plot_var_1) ,fontsize=10)
plt.ylabel('{}'.format(plot_var_2) ,fontsize=10)
plt.xticks(np.arange(tick_min, tick_max+0.1, 20))    
plt.title('{}'.format(plot_title), fontsize=10)
plt.set_xlim(xmin300, 0)
ylim(ymin=1)
plt.grid(True)
plt.show()     
'''
df[df['pred_depvar1']>=0.75]
len(df[df['pred_depvar1']>=0.75])



