#%%
#python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math 
import os 

os.getcwd()
os.chdir('D:\\MyDrive\\10. MS in Data Science UofWisconsin\\14. Final Project Maestria')

#Excel file con nombres de los stocks
df_tickers=pd.read_excel('./Predicting Stocks/Time Series/marketInsider.xlsx',sheet_name= 'automatically') #current 30 stocks of DJI
tickerList=df_tickers.ticker.to_list()
tickerList=[x.strip() for x in tickerList]



degrees = np.linspace(start=12,stop=360,num=30)

heights = np.sin(degrees*math.pi/180)
lengths = np.cos(degrees*math.pi/180)

unitCircle=pd.DataFrame({
    'heights': heights,
    'lengths': lengths,
    'securities': tickerList    
})

printFig = 0 

if printFig == 1: 
    plt.scatter(x=unitCircle.lengths, y=unitCircle.heights, s=5, color='b')
    # Add labels
    for i, txt in enumerate(unitCircle['securities']):
        plt.annotate(txt, (unitCircle['lengths'][i], unitCircle['heights'][i]))

    plt.annotate('P1',(1,1.3))
    plt.annotate('DJI', (0, 0))
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5,1.5)
    plt.title('Unit Circle')
    plt.show()




# %%
