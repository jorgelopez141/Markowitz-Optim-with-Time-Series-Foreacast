#%%
import numpy as np 

unif_rand_n=np.random.uniform(low=1,high=100,size=4)
rand_nums=unif_rand_n/unif_rand_n.sum()

rand_nums,rand_nums.sum()