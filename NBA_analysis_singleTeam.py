import json
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics, cross_validation
import time

#plt.rcParams['axes.facecolor'] = 'black'

# function to lookup team_id given team_abbreviation
# input paramters:
# tm_dict  --> dictionary with team_ids and team_abbreviations, one id can have multiple abbreviations
# team_abv --> team abbreviation
# output:
# key --> team id
def lookup_team_id(tm_dict,team_abv):
    for key,values in tm_dict.items():
        for value in values:
            if value == team_abv:
                return key

# function to get X, y data of a team against its opponent team for logistic regression.
# input parameters:
# dframe    --> dataframe with all the data
# teamID    --> team id of the team you want the data for
# oppTeamID --> team id of the opponent team you want the data for
# outputs:
# X --> feature matrix
# y --> outcome vector, win (1) or loss (0) in this case
def prepare_data(dframe, teamID, oppTeamID):
    data_all_teamID = dframe.loc[df['TEAM_ID'] == teamID].loc[df['OPPONENT_ID'] == oppTeamID]
    X = data_all_teamID[['WL','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PF']].copy()
    y = X['WL'].copy()
    # set the first column of X to be an intercept column and set its value to 1
    X['WL'] = 1 # intercept column
    rnd_indx = np.random.permutation(y.index.size)
    X = np.array(X.iloc[rnd_indx])
    y = np.array(y.iloc[rnd_indx])
    return X, y


###################################################################################################################

#start_time = time.time()

# List of teams available:
#
# [ ATL, BOS, CLE, NOH, CHI, DAL, DEN, GOS, HOU, LAC, LAL, MIA, MIL, MIN, NJN,
#   NYK, ORL, IND, PHL, PHX, POR, SAC, SAN, SEA, TOR, UTH, VAN, WAS, DET, CHH ]

team    = 'CHI'  # Chicago Bulls
oppTeam = 'BOS'  # Boston Celtics

# read gameLog data from json files and convert it to dataframe.
df = pd.DataFrame()
yr = 1985 # starting year (data available from 1985 - 2016)
for i in range(0,32):
    yr_str = str(yr)+'-'+str(yr+1)[2:]
    yr += 1
    with open('Data/gamelog/gameLog_'+yr_str+'.json') as json_data:
        d = json.load(json_data)
        headers = d['resultSets'][0]['headers']
        df_tmp = pd.DataFrame(d['resultSets'][0]['rowSet'], columns=headers)
        df = df.append(df_tmp)


# cleaning data
# replace '@' by 'vs.' in column matchup
df['MATCHUP'] = df['MATCHUP'].str.replace('@','vs.')

# drop columns with missing values
df = df.dropna()

# rearrange the index numbers
df.index = np.linspace(0,df.index.size-1,df.index.size)

# replace 'W' and 'L' to 1 and 0 in column WL
df.WL.replace(['W','L'],[1,0], inplace=True)

# save team_id and team_abbreviations (some teams have multiple abbreviations) on a dictionary
tm_id_nm = {k:df['TEAM_ABBREVIATION'].drop_duplicates().tolist() for k,df in df[['TEAM_ID','TEAM_ABBREVIATION']].drop_duplicates().groupby('TEAM_ID')}

teamID = lookup_team_id(tm_id_nm, team)
oppTeamID = lookup_team_id(tm_id_nm, oppTeam)
#print teamID, oppTeamID

# get team_id of the teams to add a new column into the dataframe
matchup_team_ids = np.zeros(len(df.index))
for i in range(0,len(df.index)):
    matchup_team_ids[i] = lookup_team_id(tm_id_nm,df['MATCHUP'][i][-3:])

# add a new column of opponent_id to the dataframe
df['OPPONENT_ID'] = matchup_team_ids

# sort data by TEAM_ID, OPPONENT_ID and GAME_DATE
df = df.sort_values(['TEAM_ID', 'OPPONENT_ID','WL'])

# count the number of games played by each team
gameCount_byTeam = df.TEAM_ID.value_counts().sort_index()

# get the team number (determined alphabetically): should be --> 0 <= team_num <= 29
team_one = lookup_team_id(tm_id_nm, 'ATL')
if teamID == team_one:
    team_num=0
else:
    team_num = lookup_team_id(tm_id_nm, team) - team_one

loss_count = 0
win_count = 0
tmp = np.zeros(gameCount_byTeam.index.size+1).astype(int)
tmp[1:] = gameCount_byTeam.tolist()

# get the starting and ending index to look up the games of the team chosen
# it is used to get the number of games played with the opponent teams
strt_ind = 0
end_ind = tmp[1]
if team_num != 0:
    for i in range(0,team_num):
        strt_ind += tmp[i+1]
        end_ind += tmp[i+2]

# count the number of games played by the team with all its opponent teams
gameCount_ofTeam_byOpponent = pd.Series(df['OPPONENT_ID'].tolist()[strt_ind:end_ind]).value_counts().sort_index()

# get the starting and ending index to look up the games of the team with its opponent teams
tmp1 = np.zeros(gameCount_ofTeam_byOpponent.index.size+1).astype(int)
tmp1[1:] = gameCount_ofTeam_byOpponent.tolist()
start_indx = 0
end_indx = tmp1[1]

# dictionary and array to store the results
winLoss_records_byTeam = {}
winLoss_records = []

# get the number of wins and losses of the team against its opponent teams
for j in range(0,gameCount_ofTeam_byOpponent.index.size):
    winLoss_count = pd.Series(df['WL'].tolist()[start_indx:end_indx]).value_counts().sort_index()
    # check if the team has lost all the games or won all the games or lost some and won some
    if 0 not in winLoss_count.index.tolist():
        loss_count = 0
        win_count = winLoss_count.tolist()[0]
    elif 1 not in winLoss_count.index.tolist():
        win_count = 0
        loss_count = winLoss_count.tolist()[0]
    else:
        loss_count = winLoss_count[0]
        win_count = winLoss_count[1]
    # stop looking for start and end indices after the last opponent team
    if j+2 != gameCount_ofTeam_byOpponent.index.size+1:
        start_indx += tmp1[j+1]
        end_indx += tmp1[j+2]
    # append the wins and losses of each team in a 2d array
    winLoss_records.append(np.array([gameCount_ofTeam_byOpponent.index[j].astype(int), loss_count, win_count]))

# store the results in a dictionary
winLoss_records_byTeam[teamID] = np.array(winLoss_records)

#print winLoss_records_byTeam

###########################################################################
# saving data and plotting
#df.to_csv("gameLog_1985-2017_1.csv", sep=',')

#print 'Execution time: ', time.time()-start_time

############ plot 1
# plot a bar graph of win/loss percentage of the team against all its opponent teams
df2 = pd.DataFrame(winLoss_records_byTeam[teamID][:,1:],columns=['Loss','Win'])
print(df2['Loss'])
df2['Loss'] = (df2['Loss']/gameCount_ofTeam_byOpponent.tolist())*100
df2['Win'] = (df2['Win']/gameCount_ofTeam_byOpponent.tolist())*100
df3 = []
# get TEAM Abbreviations of the opponent teams for x-axis
for i in range(0,df2.index.size+1):
    if i != team_num:
        a = tm_id_nm[gameCount_byTeam.index.tolist()[i]][0]
        df3.append(a)
# change the idex of the dataframe to team abbreviation
df2.index = df3
df2.plot.bar()
plt.xlabel('Opponent Teams')
plt.ylabel('Percentage of Wins and Losses')
plt.title('Win/Loss record for '+team+' (1985-Present)')

########### plot 2
# plot Field Goal percentages of the team against an opponent teams
OPP_WIN = df.loc[df['TEAM_ID'] == teamID].loc[df['OPPONENT_ID'] == oppTeamID].loc[df['WL'] == 1].sort_values('GAME_DATE')
OPP_LOSS = df.loc[df['TEAM_ID'] == teamID].loc[df['OPPONENT_ID'] == oppTeamID].loc[df['WL'] == 0].sort_values('GAME_DATE')
OPP_WIN_dates = mp.dates.date2num(pd.to_datetime(OPP_WIN['GAME_DATE']).tolist())
OPP_LOSS_dates = mp.dates.date2num(pd.to_datetime(OPP_LOSS['GAME_DATE']).tolist())

plt.figure(2)
plt.plot_date(OPP_WIN_dates,OPP_WIN['FG_PCT']*100,'o--')
plt.plot_date(OPP_LOSS_dates,OPP_LOSS['FG_PCT']*100,'o--')
plt.plot((np.min(OPP_WIN_dates),np.max(OPP_WIN_dates)),(OPP_WIN['FG_PCT'].mean()*100,OPP_WIN['FG_PCT'].mean()*100), 'b--')
plt.plot((np.min(OPP_LOSS_dates),np.max(OPP_LOSS_dates)),(OPP_LOSS['FG_PCT'].mean()*100,OPP_LOSS['FG_PCT'].mean()*100), 'g--')
plt.xlabel('GAME_DATE')
plt.ylabel('Field Gaol Percentage')
plt.title(team+' vs. '+oppTeam+' (1985 - Present)')
plt.legend(['Win','Loss'])
plt.show(block=False)


#############################################################################################################
############################ Logistic regression

# prepare data for logistic regression
# X --> feature matrix
# y --> binary category (1 or 0)
X, y = prepare_data(df,teamID,oppTeamID)
#print X

# instantiate a Logistic regression model, and fit with X and y
lrModel = LogisticRegression()
lrModel = lrModel.fit(X,y)

# check the accuracy on the training set
print('')
print( '################# Logistic Regression report ####################')

# evaluate the model by splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
lrModel2 = LogisticRegression()
lrModel2.fit(X_train, y_train)

# predict class labels from the test set
predicted = lrModel2.predict(X_test)
print('')
print( 'Actual classes of the test data   : ', y_test)
print( '')
print( 'Predicted classes of the test data: ', predicted)
print( '')

# generate class probabilities
probs = lrModel2.predict_proba(X_test)
#print probs

cm = metrics.confusion_matrix(y_test, predicted)
print( 'Confusion matrix')
print( 'Class labels: 0     1')
print( '             ', cm[0][0], '  ', cm[0][1])
print( '             ', cm[1][0], '  ', cm[1][1])

print( '')
print( 'Classification report:')
print( metrics.classification_report(y_test, predicted))
print( '')

# evaluate the model using 10-fold cross-validation
num_folds = 10
num_instances = len(X)
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
scoring = 'accuracy'
scores = cross_val_score(LogisticRegression(), X, y, scoring=scoring, cv=kfold)
print('10-fold cross-validation accuracy: %.3f (%.3f)' % (scores.mean(), scores.std()))

# predicting the probability of winning
#features = np.array([1, 48, 86, 1, 2, 13, 27, 10, 27, 37, 13, 7, 20, 22])
#print lrModel.predict_proba(features.reshape(1,-1))
