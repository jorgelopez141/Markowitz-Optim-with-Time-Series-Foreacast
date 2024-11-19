#%%
import numpy as np

def random_gen_weights(n):
    laCadena = np.zeros(n)  # Create a vector of zeroes
    
    for i in range(n):
        if i == 0:
            secuencia = np.arange(0, 1.01, 0.01)  # Create a sequence from 0 to 1 by 0.01
            x = np.random.choice(secuencia, 1)[0]
            laCadena[i] = x
            if np.sum(laCadena) == 1:
                break
        elif i < n - 1:
            secuencia = np.arange(0, 1 - np.sum(laCadena) , 0.01)
            if len(secuencia) >= 1 and np.sum(laCadena) < 1:
                x = np.random.choice(secuencia, 1)[0]
                laCadena[i] = x
                if np.sum(laCadena) == 1:
                    break
            else:
                break
        else:
            if np.sum(laCadena) < 1:
                laCadena[i] = 1 - np.sum(laCadena)
    
    np.random.shuffle(laCadena)  # Shuffle the vector to randomize the order
    return laCadena




#%%


#%%

