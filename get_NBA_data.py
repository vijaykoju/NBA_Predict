#import requests
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup
import urllib2

#leaguedashplayerstats_url = "http://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight="

#yr = "1985-86"
## need to get data from 1985-86 to 2016-17
##gameLog_url = "http://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&LeagueID=00&PlayerOrTeam=T&Season=1985-86&SeasonType=Regular+Season&Sorter=PTS"
#gameLog_url = "http://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&LeagueID=00&PlayerOrTeam=T&Season="+yr+"&SeasonType=Regular+Season&Sorter=PTS"
#
#response = requests.get(gameLog_url)
#response.raise_for_status()
#gameLog = response.json()['resultSets'][0]['rowSet']
#
#print len(gameLog) 
#
#print gameLog[0]
#
#headers = response.json()['resultSets'][0]['headers']
#
#df = pd.DataFrame(gameLog, columns=headers)
#
#df.to_csv("gameLog_"+yr+".csv", sep=',')
#
##plt.plot(df['BLK'])
##plt.show()


yr = ["1985-86","1986-87","1987-88"]
# need to get data from 1985-86 to 2016-17
#gameLog_url = "http://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&LeagueID=00&PlayerOrTeam=T&Season=1985-86&SeasonType=Regular+Season&Sorter=PTS"

for year in yr:
    gameLog_url = "http://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&LeagueID=00&PlayerOrTeam=T&Season="+year+"&SeasonType=Regular+Season&Sorter=PTS"
    url = urllib2.urlopen(gameLog_url)
    data = url.read()
    soup = BeautifulSoup(data)
    newData = json.loads(str(soup))
    print newData
    #print gameLog_url
    #
    #s = requests.Session()
    #response = s.get(gameLog_url)
    ##response.raise_for_status()
    #print year+'done.\n\n\n\n\n\n'
    #print response.text, '\n\n\n\n'
    ##gameLog = response.json()['resultSets'][0]['rowSet']
    #
    #print len(gameLog) 
    #
    #print gameLog[0]
    
    #headers = response.json()['resultSets'][0]['headers']
    #print headers

#df = pd.DataFrame(gameLog, columns=headers)
#
#df.to_csv("gameLog_"+yr+".csv", sep=',')
