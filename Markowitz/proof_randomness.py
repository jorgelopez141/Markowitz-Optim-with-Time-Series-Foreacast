#%%
from numbersAddTo1 import random_gen_weights
from unitCircle import unitCircle as unitCircleDf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

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


plt.scatter(x=unitCircleDf.lengths, y=unitCircleDf.heights, s=5, color='b')
# Add labels
for i, txt in enumerate(unitCircleDf['securities']):
    plt.annotate(txt, (unitCircleDf['lengths'][i], unitCircleDf['heights'][i]))

plt.scatter(x=rand_df_1.rand_length, y=rand_df_1.rand_height, s=1, color='r')
plt.scatter(x=rand_df_unif.rand_length, y=rand_df_unif.rand_height, s=1, color='black')
plt.annotate('DJI', (0, 0))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5,1.5)
plt.title('Unit Circle')
plt.show()
