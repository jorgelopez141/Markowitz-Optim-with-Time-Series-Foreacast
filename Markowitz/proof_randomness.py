#%%
from numbersAddTo1 import random_gen_weights
from unitCircle import unitCircle as unitCircleDf
#Roger Stafford (2024). Random Vectors with Fixed Sum (https://www.mathworks.com/matlabcentral/fileexchange/9700-random-vectors-with-fixed-sum), MATLAB Central File Exchange. Retrieved October 13, 2024.

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import os

# Change to the directory of the current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from randomness_by_RogerStafford import randfixedsum 

n_rows = 10000
n_cols = 30
data = []
data_uniform= []

for _ in range(n_rows):
    row = random_gen_weights(n_cols)
    unif_rand_n=np.random.uniform(low=1,high=100,size=n_cols)
    row_unif = unif_rand_n/unif_rand_n.sum()
    data.append(row)
    data_uniform.append(row_unif)

df_weights = pd.DataFrame(data)
df_weights_unif = pd.DataFrame(data_uniform)

rand_RS,v = randfixedsum(30, 10000, 1, 0, 1)
df_weights_RS = pd.DataFrame(rand_RS.T)

#%%

from siuba import * 
from plotnine import * 

#%%

df_names=pd.read_excel('marketInsider.xlsx',sheet_name='automatically')

#%%

df_weights_unif.columns = df_names['ticker']

distro_uniform= df_weights_unif>> select(_[:10]) >> \
gather('key','value') >> ggplot() + geom_boxplot(aes(x='key',y='value', color='key'))+\
theme_dark() + \
theme(text=element_text(family='sans-serif',size=14),
      plot_title=element_text(hjust=0, family='sans-serif', size=14),
        plot_subtitle=element_text(hjust=0, family='sans-serif', size=14, style='italic',margin={'t': 20})
    )

distro_uniform

#%%

#gt1._options.container_padding_y 
distro_uniform.save(filename = 'Distro Uniform', height=8, width=12, units = 'in', dpi=1000)

#%%
df_weights_RS.columns = df_names['ticker']

RogerStaffordDistro=df_weights_RS>> select(_[:10]) >> \
gather('key','value') >> ggplot() + geom_boxplot(aes(x='key',y='value', color='key'))+\
theme_dark() + \
theme(text=element_text(family='sans-serif',size=14)      
    )

RogerStaffordDistro.save(filename = 'Roger Stafford Distro', height=8, width=12, units = 'in', dpi=1000)

#%%
from siuba.dply.vector import * 

df_weights_RS >> gather('key','value') >> \
group_by(_.key) >> summarize(mean_value = _.value.mean(), median_value = _.value.median(), 
                             max_value = _.value.max(),
                             min_value = _.value.min()
                             
                             ) >> ungroup() >> _.describe()




#%%
df_weights.columns = df_names['ticker']

LopezDistro = df_weights>> select(_[:10]) >> \
gather('key','value') >> ggplot() + geom_boxplot(aes(x='key',y='value', color='key'))+\
theme_dark() + \
theme(text=element_text(family='sans-serif',size=14)      
    )

LopezDistro.save(filename = 'Lopez Distro', height=8, width=12, units = 'in', dpi=1000)

#%%

df_weights >> gather('key','value') >> \
group_by(_.key) >> summarize(mean_value = _.value.mean(), median_value = _.value.median(), 
                             max_value = _.value.max(),
                             min_value = _.value.min()
                             
                             ) >> ungroup() >> _.describe()

#%%
df_weights_unif >> gather('key','value') >> \
group_by(_.key) >> summarize(mean_value = _.value.mean(), median_value = _.value.median(), 
                             max_value = _.value.max(),
                             min_value = _.value.min()
                             
                             ) >> ungroup() >> _.describe()

#%%

#%%

# original unit circle
plt.scatter(x=unitCircleDf.lengths, y=unitCircleDf.heights, s=5, color='b')
# Add labels
for i, txt in enumerate(unitCircleDf['securities']):
    plt.annotate(txt, (unitCircleDf['lengths'][i], unitCircleDf['heights'][i]))

plt.annotate('P1',(1,1.3))
plt.annotate('DJI', (0, 0))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5,1.5)
plt.title('Unit Circle')
plt.show()

#%%

from function_filePrep import tickerList, download_data,missing_days_andIndexTimeZone, to_month_and_add_monthYear_columns

#%%
df_ticker_price = download_data(list_stocks=tickerList,start_date = '2023-08-01', end_date = '2024-09-30')
df_ticker_price1=missing_days_andIndexTimeZone(df_ticker_price)

#%%
# top 10 companies by weight as of sept 27, 2024
df_top10_byWeight=df_ticker_price1.apply(lambda x: x/sum(x), axis=1) >> filter(_.index == '2024-09-27') >> gather() >> \
arrange(-_.value) >> head(10)

#%%
# box plot of weights in time of biggest 10 securities (price) 

bx_big10Sec=df_ticker_price1.apply(lambda x: x/sum(x), axis=1) >>\
     gather(key='key', value='value') >> \
     inner_join(_,df_top10_byWeight, by = {'key':'key'})>> ggplot() + geom_boxplot(aes(x='key',y='value_x', color='key')) + \
     theme(text=element_text(family='sans-serif',size=14),
      plot_title=element_text(hjust=0, family='sans-serif', size=14),
        plot_subtitle=element_text(hjust=0, family='sans-serif', size=14, style='italic',margin={'t': 20})
    ) +theme_dark()

#gt1._options.container_padding_y 
bx_big10Sec.save(filename = 'Box Plot Weights Big 10 Sec', height=5, width=8, units = 'in', dpi=1000)


#%%
# box plot of weights in first 10 securities 

df_ticker_price1.apply(lambda x: x/sum(x), axis=1) >>\
     select(_[:10]) >> gather(key='key', value='value') >> ggplot() + geom_boxplot(aes(x='key',y='value', color='key')) + \
     theme(text=element_text(family='sans-serif',size=14),
      plot_title=element_text(hjust=0, family='sans-serif', size=14),
        plot_subtitle=element_text(hjust=0, family='sans-serif', size=14, style='italic',margin={'t': 20})
    )

#%%

# recent weights DJI 
weights_DJI_recent=(df_ticker_price1.apply(lambda x: x/sum(x), axis=1) >> filter(_.index == '2024-09-27')).iloc[0,:]

#%%

y_recentDJI=np.array(weights_DJI_recent).dot(unitCircleDf['heights'])
x_recentDJI=np.array(weights_DJI_recent).dot(unitCircleDf['lengths'])

#%%

unitCircle_init=unitCircleDf >> ggplot()+ \
geom_text(aes(x='lengths',y='heights',label='securities'))+\
geom_point(pd.DataFrame({'x':[0, x_recentDJI],'y':[0,y_recentDJI],'Portfolios': ['Eq. Weight','DJI']}),aes(x='x', y='y', color='Portfolios')) + \
geom_text(pd.DataFrame({'x':[-.1],'y':[0],'theLabel': ['Eq. W.']}),aes(x='x', y='y', label='theLabel')) + \
geom_text(pd.DataFrame({'x':[.15],'y':[0],'theLabel': ['DJI']}),aes(x='x', y='y', label='theLabel')) + \
labs(x='',y='')+ \
    theme(text=element_text(family='sans-serif',size=14), 
          legend_position= 'bottom',
          axis_line=element_blank(),
          axis_text=element_blank(),
          axis_ticks=element_blank()
        ) 
unitCircle_init.save(filename = 'Init Unit Circle', height=6,width=6, units = 'in', dpi=1000)

#%%
### los 3 dataframes 
### df_weights, df_weights_unif , df_weights_RS 

h_Lopez=[row.dot(unitCircleDf['heights']) for row in df_weights.values]
w_Lopez=[row.dot(unitCircleDf['lengths']) for row in df_weights.values]

h_unif=[row.dot(unitCircleDf['heights']) for row in df_weights_unif.values]
w_unif=[row.dot(unitCircleDf['lengths']) for row in df_weights_unif.values]

h_Stafford=[row.dot(unitCircleDf['heights']) for row in df_weights_RS.values]
w_Stafford=[row.dot(unitCircleDf['lengths']) for row in df_weights_RS.values]

Lopez_rnd=pd.DataFrame({
    'height': h_Lopez,
    'width': w_Lopez,
    'method': 'Lopez'
})

Unif_rnd = pd.DataFrame({
    'height': h_unif,
    'width': w_unif,
    'method': 'Uniform'
})

Stafford_rnd = pd.DataFrame({
    'height': h_Stafford,
    'width': w_Stafford,
    'method': 'Stafford'
})

all_methods_random = pd.concat([Lopez_rnd,Stafford_rnd, Unif_rnd ])

#%%

all_methods_random >> 

#%%


random_method_unitCircle=unitCircleDf >> ggplot()+ \
geom_text(aes(x='lengths',y='heights',label='securities'))+\
geom_point(all_methods_random,aes(x='width', y='height', color='method'),  show_legend=True) +\
scale_color_manual(
    name='\n',
    values={'Lopez': '#3C1742', 'Stafford': '#F3FFB9', 'Uniform': '#C42021'},
    labels={'Lopez': 'Lopez', 'Stafford': 'Stafford', 'Uniform': 'Uniform'}
) + \
labs(x='',y='')+ \
    theme(text=element_text(family='sans-serif',size=14), 
          legend_position= 'bottom',
          axis_line=element_blank(),
          axis_text=element_blank(),
          axis_ticks=element_blank()
        ) 

random_method_unitCircle.save(filename = 'Rnd Methods Unit Circle', height=6,width=6, units = 'in', dpi=1000)


#%%

unitCircleDf >> ggplot()+ \
geom_text(aes(x='lengths',y='heights',label='securities'))+\
geom_point(Lopez_rnd,aes(x='width', y='height'), color='#3C1742', show_legend=True) + \
geom_point(Stafford_rnd,aes(x='width', y='height'), color='#F3FFB9',show_legend=True) + \
geom_point(Unif_rnd,aes(x='width', y='height'), color= '#C42021',show_legend=True) + \
scale_color_manual(
    name='Legend Title',
    values={'Lopez': '#3C1742', 'Stafford': '#F3FFB9', 'Uniform': '#C42021'},
    labels={'Lopez': 'Lopez rnd', 'Stafford': 'Stafford rnd', 'Uniform': 'Unif rnd'}
) + guides(color=guide_legend(override_aes={'alpha': 1})) + \
theme(
    text=element_text(family='sans-serif', size=14),
    plot_title=element_text(hjust=0, family='sans-serif', size=14),
    plot_subtitle=element_text(hjust=0, family='sans-serif', size=14, 
                               style='italic', margin={'t': 20})
)




#%%

# WITH RANDOM WEIGHTS
random_lengths=[np.array(x).dot(unitCircleDf.lengths) for x in df_weights.values]
random_heights=[np.array(x).dot(unitCircleDf.heights) for x in df_weights.values]


rand_df_1=pd.DataFrame({
    'rand_length': random_lengths,
    'rand_height': random_heights
})

rand_df_unif = pd.DataFrame(
    {
        'rand_length': [np.array(x).dot(unitCircleDf.lengths) for x in df_weights_unif.values],
        'rand_height': [np.array(x).dot(unitCircleDf.heights) for x in df_weights_unif.values]
    }
)
#df_weights_RS

rand_df_RS = pd.DataFrame(
    {
        'rand_length': [np.array(x).dot(unitCircleDf.lengths) for x in df_weights_RS.values],
        'rand_height': [np.array(x).dot(unitCircleDf.heights) for x in df_weights_RS.values]
    }
)


plt.scatter(x=unitCircleDf.lengths, y=unitCircleDf.heights, s=5, color='b')
# Add labels
for i, txt in enumerate(unitCircleDf['securities']):
    plt.annotate(txt, (unitCircleDf['lengths'][i], unitCircleDf['heights'][i]))


plt.scatter(x=rand_df_1.rand_length, y=rand_df_1.rand_height, s=1, color='r', label = 'Lopez')
plt.scatter(x=rand_df_RS.rand_length, y=rand_df_RS.rand_height, s=1, color='blue', label = 'Stafford')
plt.scatter(x=rand_df_unif.rand_length, y=rand_df_unif.rand_height, s=1, color='black', label = 'Naive')


plt.annotate('DJI', (0, 0))
# Adding legend with bigger dots
plt.legend(markerscale=5)    



plt.xlim(-1.5, 1.5)
plt.ylim(-1.5,1.5)
plt.title('Unit Circle')
plt.show()
