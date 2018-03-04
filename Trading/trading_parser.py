##Parser
'''
Description:
	Parsing the league name. ESP league is not optimal
'''

def parser_league(Dataframe):
	df = Dataframe
	league_dic={'Premier League':'ING','Primeira Liga':'PT'}
	df=df.replace({"var1_08": league_dic})

	n = 0
	for i in df.loc[:, 'var1_08']:
		if (i != 'PT' and i != 'ING'):
			#print(i)
			df.loc[n, 'var1_08'] = 'ESP'
		n += 1

	return df