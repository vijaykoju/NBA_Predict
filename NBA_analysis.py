import json
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import time

start_time = time.time()

# read gameLog data from json files and convert it to dataframe.
df = pd.DataFrame()
yr = 1985
for i in range(0,32):
    yr_str = str(yr)+'-'+str(yr+1)[2:]
    yr += 1
    with open('gameLog_'+yr_str+'.json') as json_data:
        d = json.load(json_data)
        headers = d['resultSets'][0]['headers']
        df_tmp = pd.DataFrame(d['resultSets'][0]['rowSet'], columns=headers)
        df = df.append(df_tmp) 

# sort data by TEAM_ID and SEASON_ID
#df = df.sort_values(['TEAM_ID', 'SEASON_ID'])
df = df.sort_values(['TEAM_ID', 'MATCHUP','WL'])

# replace '@' by 'vs.' in column matchup
df['MATCHUP'] = df['MATCHUP'].str.replace('@','vs.')

# drop columns with missing values
df = df.dropna()

# rearrange the index numbers
df.index = np.linspace(0,df.index.size-1,df.index.size)

# replace 'W' and 'L' to 1 and 0 in column WL
df.WL.replace(['W','L'],[1,0], inplace=True)

# save team_id and team_abbreviations (some teams have multiple abbreviations) on a dict
tm_id_nm = {k:df['TEAM_ABBREVIATION'].drop_duplicates().tolist() for k,df in df[['TEAM_ID','TEAM_ABBREVIATION']].drop_duplicates().groupby('TEAM_ID')}

# function to lookup team_id given team_abbreviation
def lookup_team_id(tm_dict,team_abv):
    for key,values in tm_dict.items():
        for value in values:
            if value == team_abv:
                return key

results = np.zeros(len(df.index))
for i in range(0,len(df.index)):
    results[i] = lookup_team_id(tm_id_nm,df['MATCHUP'][i][-3:])

df['OPPONENT_ID'] = results
df = df.sort_values(['TEAM_ID', 'OPPONENT_ID','WL'])


winLoss_records_byTeam = {}
# counting occurances of distinct items in a column 
gameCount_byTeam = df.TEAM_ID.value_counts().sort_index()

loss_count = 0
win_count = 0
tmp = np.zeros(gameCount_byTeam.index.size+1).astype(int)
tmp[1:] = gameCount_byTeam.tolist()
strt_ind = 0 
end_ind = tmp[1] 
for i in range(0,gameCount_byTeam.index.size):
    gameCount_ofTeam_byOpponent = pd.Series(df['OPPONENT_ID'].tolist()[strt_ind:end_ind]).value_counts().sort_index()
    if i+2 != gameCount_byTeam.index.size+1:
        strt_ind += tmp[i+1]
        end_ind += tmp[i+2]
    tmp1 = np.zeros(gameCount_ofTeam_byOpponent.index.size+1).astype(int)
    tmp1[1:] = gameCount_ofTeam_byOpponent.tolist()
    start_indx = 0
    end_indx = tmp1[1]
    winLoss_records = []
    for j in range(0,gameCount_ofTeam_byOpponent.index.size):
        winLoss_count = pd.Series(df['WL'].tolist()[start_indx:end_indx]).value_counts().sort_index()
        if 0 not in winLoss_count.index.tolist():
            loss_count = 0
            win_count = winLoss_count.tolist()[0]
        elif 1 not in winLoss_count.index.tolist():
            win_count = 0
            loss_count = winLoss_count.tolist()[0]
        else:
            loss_count = winLoss_count[0]
            win_count = winLoss_count[1]

        if j+2 != gameCount_ofTeam_byOpponent.index.size+1:
            start_indx += tmp1[j+1]
            end_indx += tmp1[j+2]
        winLoss_records.append(np.array([gameCount_ofTeam_byOpponent.index[j].astype(int), loss_count, win_count]))
    winLoss_records_byTeam[gameCount_byTeam.index[i].astype(int)] = np.array(winLoss_records)

#print winLoss_records_byTeam

###########################################################################
# saving data and plotting
#df.to_csv("gameLog_1985-2017_1.csv", sep=',')

print 'Execution time: ', time.time()-start_time

# must be: 0 <= team_num <= 28
team_num=28
team_name='determined below'

strt_ind = 0
end_ind = tmp[1] 
if team_num != 0:
    for i in range(0,team_num):
        strt_ind += tmp[i+1]
        end_ind += tmp[i+2]
gameCount_ofTeam_byOpponent = pd.Series(df['OPPONENT_ID'].tolist()[strt_ind:end_ind]).value_counts().sort_index()

# plot 1
df2 = pd.DataFrame(winLoss_records_byTeam[gameCount_byTeam.index[team_num].astype(int)][:,1:],columns=['Loss','Win'])
df2['Loss'] = (df2['Loss']/gameCount_ofTeam_byOpponent.tolist())*100
df2['Win'] = (df2['Win']/gameCount_ofTeam_byOpponent.tolist())*100
#df2.index = gameCount_byTeam.index.tolist()[1:]
df3 = []
for i in range(0,df2.index.size+1):
    if i != team_num:
        a = tm_id_nm[gameCount_byTeam.index.tolist()[i]][0]
        df3.append(a)
    else:
        team_name = tm_id_nm[gameCount_byTeam.index.tolist()[i]][0]
df2.index = df3
df2.plot.bar()
plt.xlabel('Opponent Teams')
plt.ylabel('Percentage of Wins and Losses')
plt.title('Win/Loss record for '+team_name+' (1985-Present)')

# plot 2
SAC_teamID = lookup_team_id(tm_id_nm,'SAC')
SAC_WIN = df.loc[df['TEAM_ID'] == gameCount_byTeam.index[team_num].astype(int)].loc[df['OPPONENT_ID'] == SAC_teamID].loc[df['WL'] == 1].sort_values('GAME_DATE')
SAC_LOSS = df.loc[df['TEAM_ID'] == gameCount_byTeam.index[team_num].astype(int)].loc[df['OPPONENT_ID'] == SAC_teamID].loc[df['WL'] == 0].sort_values('GAME_DATE')
SAN_teamID = lookup_team_id(tm_id_nm,'SAN')
SAN_WIN = df.loc[df['TEAM_ID'] == gameCount_byTeam.index[team_num].astype(int)].loc[df['OPPONENT_ID'] == SAN_teamID].loc[df['WL'] == 1].sort_values('GAME_DATE')
SAN_LOSS = df.loc[df['TEAM_ID'] == gameCount_byTeam.index[team_num].astype(int)].loc[df['OPPONENT_ID'] == SAN_teamID].loc[df['WL'] == 0].sort_values('GAME_DATE')
SAC_WIN_dates = mp.dates.date2num(pd.to_datetime(SAC_WIN['GAME_DATE']).tolist())
SAC_LOSS_dates = mp.dates.date2num(pd.to_datetime(SAC_LOSS['GAME_DATE']).tolist())
SAN_WIN_dates = mp.dates.date2num(pd.to_datetime(SAN_WIN['GAME_DATE']).tolist())
SAN_LOSS_dates = mp.dates.date2num(pd.to_datetime(SAN_LOSS['GAME_DATE']).tolist())
plt.figure(2)
plt.subplot(211)
plt.plot_date(SAC_WIN_dates,SAC_WIN['FG_PCT']*100,'o--')
plt.plot_date(SAC_LOSS_dates,SAC_LOSS['FG_PCT']*100,'o--')
plt.plot((np.min(SAC_WIN_dates),np.max(SAC_WIN_dates)),(SAC_WIN['FG_PCT'].mean()*100,SAC_WIN['FG_PCT'].mean()*100), 'b--')
plt.plot((np.min(SAC_LOSS_dates),np.max(SAC_LOSS_dates)),(SAC_LOSS['FG_PCT'].mean()*100,SAC_LOSS['FG_PCT'].mean()*100), 'g--')
plt.xlabel('GAME_DATE')
plt.ylabel('Field Gaol Percentage')
plt.title(team_name+' vs. SAC (1985 - Present)')
plt.legend(['Win','Loss'])
plt.subplot(212)
plt.plot_date(SAN_WIN_dates,SAN_WIN['FG_PCT']*100,'o--')
plt.plot_date(SAN_LOSS_dates,SAN_LOSS['FG_PCT']*100,'o--')
plt.plot((np.min(SAN_WIN_dates),np.max(SAN_WIN_dates)),(SAN_WIN['FG_PCT'].mean()*100,SAN_WIN['FG_PCT'].mean()*100), 'b--')
plt.plot((np.min(SAN_LOSS_dates),np.max(SAN_LOSS_dates)),(SAN_LOSS['FG_PCT'].mean()*100,SAN_LOSS['FG_PCT'].mean()*100), 'g--')
plt.xlabel('GAME_DATE')
plt.ylabel('Field Gaol Percentage')
plt.title(team_name+' vs. SAN (1985 - Present)')
plt.legend(['Win','Loss'])
plt.show(block=False)
