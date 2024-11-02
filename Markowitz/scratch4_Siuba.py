#%%
import pandas as pd 
import numpy as np 
from siuba import _, filter, arrange, select, mutate, group_by, summarize 
from siuba.data import cars 
from plotnine import *


#%%

pctiles=np.percentile(cars.hp, range(20,120,20))

#%%

pctile_groups = []
for i,x in enumerate(pctiles): 
    if i == 0: 
        pctile_groups.append(
            f"[{str(i)+'-'+str(x)}]"
         )
    else:
        pctile_groups.append(
            f"({str(pctiles[i-1])+'-'+str(round(x,2))}]"
         )
        
pctile_dict=dict(zip(pctiles,pctile_groups))

#%%

cars >> mutate(hp_cat=pd.cut(cars.hp, range(0,1000,20)))>>\
group_by(_.hp_cat, _.cyl) >> summarize(total_hp=_.hp.sum())>>\
    ggplot()+\
        geom_col(aes(x='hp_cat',y='total_hp'))+\
            facet_wrap("~cyl",nrow=2)+\
            theme(axis_text_x=element_text(rotation=45))+\
            labs(title='ABC')

#%% 

