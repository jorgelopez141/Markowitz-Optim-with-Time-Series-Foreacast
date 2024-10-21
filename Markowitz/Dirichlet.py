#%%
import numpy as np 

numbers=np.random.dirichlet(np.ones(30),size=10000)



import os 
os.chdir(r"D:\MyDrive\10. MS in Data Science UofWisconsin\14. Final Project Maestria\Markowitz")
from unitCircle import unitCircle as unitCircleDf

#%%
import matplotlib.pyplot as plt 
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
import pandas as pd 

# WITH RANDOM WEIGHTS
random_lengths=[np.array(x).dot(unitCircleDf.lengths) for x in numbers]
random_heights=[np.array(x).dot(unitCircleDf.heights) for x in numbers]

rand_df_1=pd.DataFrame({
    'rand_length': random_lengths,
    'rand_height': random_heights
})

#%%


plt.scatter(x=unitCircleDf.lengths, y=unitCircleDf.heights, s=5, color='b')
# Add labels
for i, txt in enumerate(unitCircleDf['securities']):
    plt.annotate(txt, (unitCircleDf['lengths'][i], unitCircleDf['heights'][i]))


plt.scatter(x=rand_df_1.rand_length, y=rand_df_1.rand_height, s=1, color='r', label = 'Dirichlet')


plt.annotate('DJI', (0, 0))
# Adding legend with bigger dots
plt.legend(markerscale=5)    



plt.xlim(-1.5, 1.5)
plt.ylim(-1.5,1.5)
plt.title('Unit Circle')
plt.show()

#%%
