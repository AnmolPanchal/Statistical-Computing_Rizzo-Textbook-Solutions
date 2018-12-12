
# coding: utf-8

# In[70]:


import numpy as np 
import scipy.stats
import math
import pandas
np.random.seed(seed=10000)

a,b = 3,3
x= np.random.normal(size= a * b).reshape((a,b)) 
x[:, 0] = 1 
print(x[:5, :])
betastar = np.array([0, 1, 0.1])
e = np.random.normal(size=a)
y = np.dot(x, betastar) + e
xpinv = scipy.linalg.pinv2(x) 
betahat = np.dot(xpinv, y)
betahat1 = scipy.stats.beta(xpinv,y)
print("Estimated beta:\n", betahat)
print("Estimated beta:\n", betahat1)

