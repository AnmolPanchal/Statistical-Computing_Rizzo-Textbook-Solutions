
# coding: utf-8

# In[1]:


import math 
import pandas 
import numpy as np 
import scipy.stats 
from scipy.stats import uniform
from scipy.stats import binom
import math
from scipy.stats import norm
import matplotlib    
import matplotlib.pyplot as plt


# # Problem 2

# In[2]:


#Q.2 from HW2
def box_muller():
    u1 = random.random()
    u2 = random.random()

    t = math.sqrt((-2) * math.log(u1))
    v = 2 * math.pi * u2

    return t * math.cos(v), t * math.sin(v)


# In[3]:


from numpy import random, sqrt, log, sin, cos, pi
from pylab import show,hist,subplot,figure

# transformation function
def gaussian(u1,u2):
  z1 = sqrt(-2*log(u1))*cos(2*pi*u2)
  z2 = sqrt(-2*log(u1))*sin(2*pi*u2)
  return z1,z2

# uniformly distributed values between 0 and 1
u1 = random.rand(1000)
u2 = random.rand(1000)

# run the transformation
z1,z2 = gaussian(u1,u2)

# plotting the values before and after the transformation
figure()
subplot(221) # the first row of graphs
hist(u1)     # contains the histograms of u1 and u2 
subplot(222)
hist(u2)
subplot(223) # the second contains
hist(z1)     # the histograms of z1 and z2
subplot(224)
hist(z2)
show()
#In the first row of the graph we can see, respectively, the histograms of u1 and u2 before the transformation 
#and in the second row we can see the values after the transformation, respectively z1 and z2. 
#We can observe that the values before the transformation are distributed uniformly while the histograms of the values 
#after the transformation have the typical Gaussian shape.
#The Box-Muller transform is a method for generating normally distributed random numbers from uniformly distributed 
#random numbers. The Box-Muller transformation can be summarized as follows, suppose u1 and u2 are independent random variables
#that are uniformly distributed between 0 and 1 and let 
#then z1 and z2 are independent random variables with a standard normal distribution. Intuitively, the transformation maps 
#each circle of points around the origin to another circle of points around the origin where larger outer circles are mapped 
#to closely-spaced inner circles and inner circles to outer circles. 


# In[4]:


##Random number generatrion with Box-Muller algorithm

import math
import random
import sys
import traceback


class RndnumBoxMuller:
    M     = 10        # Average
    S     = 2.5       # Standard deviation
    N     = 10000     # Number to generate
    SCALE = N // 100  # Scale for histogram

    def __init__(self):
        self.hist = [0 for _ in range(self.M * 5)]

    def generate_rndnum(self):
        ##Generation of random nos.
        try:
            for _ in range(self.N):
                res = self.__rnd()
                self.hist[res[0]] += 1
                self.hist[res[1]] += 1
        except Exception as e:
            raise

    def display(self):
       ##showing
        try:
            for i in range(0, self.M * 2 + 1):
                print("{:>3}:{:>4} | ".format(i, self.hist[i]), end="")
                for j in range(1, self.hist[i] // self.SCALE + 1):
                    print("*", end="")
                print()
        except Exception as e:
            raise

    def __rnd(self):
   ##random integers generation.
        try:
            r_1 = random.random()
            r_2 = random.random()
            x = self.S               * math.sqrt(-2 * math.log(r_1))               * math.cos(2 * math.pi * r_2)               + self.M
            y = self.S               * math.sqrt(-2 * math.log(r_1))               * math.sin(2 * math.pi * r_2)               + self.M
            return [math.floor(x), math.floor(y)]
        except Exception as e:
            raise


if __name__ == '__main__':
    try:
        obj = RndnumBoxMuller()
        obj.generate_rndnum()
        obj.display()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


# In[5]:


#Box-Muller method
#to generate gaussian values from the numbers distributed uniformly.

import numpy as np
import matplotlib.pyplot as plt

#generate from uniform dist
np.random.seed()
N = 1000
z1 = np.random.uniform(0, 1.0 ,N)
z2 = np.random.uniform(0, 1.0 ,N)
z1 = 2*z1 - 1
z2 = 2*z2 - 1

#discard if z1**2 + z2**2 <= 1
c = z1**2 + z2**2
index = np.where(c<=1)
z1 = z1[index]
z2 = z2[index]
r = c[index]

#transformation
y1 = z1*((-2*np.log(r**2))/r**2)**(0.5)
y2 = z2*((-2*np.log(r**2))/r**2)**(0.5)

#discard outlier
y1 = y1[y1 <= 5]
y1 = y1[y1 >= -5]
y2 = y2[y2 <= 5]
y2 = y2[y2 >= -5]

#plot
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.hist(y1,bins=30,color='red')
plt.title("Histgram")
plt.xlabel("y1")
plt.ylabel("frequency")
ax2 = fig.add_subplot(2,1,2)
ax2.hist(y2,bins=30,color='blue')
plt.xlabel("y2")
plt.ylabel("frequency")
plt.show()


#  #Problem 1

# In[8]:


######################################Problem 1################################
x = binom.rvs(0, 0.3, size=1000)
y = binom.rvs(1, 0.2, size=1000)
z = binom.rvs(3, 0.5, size=1000)
print(x)
print(y)
print(z)


# In[7]:


# for inline plots in jupyter
get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(4.5,3)})
data_binom_0 = binom.rvs(n=0,p=0.3, size=1000) 

print(data_binom_0)
ax = sns.distplot(data_binom_0,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')

data_binom_1 = binom.rvs(n=1,p=0.2, size=1000) 

print(data_binom_1)
ax = sns.distplot(data_binom_1,
                  kde=False,
                  color='green',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')


data_binom_3 = binom.rvs(n=3,p=0.5, size=1000) 

print(data_binom_3)
ax = sns.distplot(data_binom_3,
                  kde=False,
                  color='red',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')


# # Problem 7

# In[8]:


import numpy as np
from scipy.stats import gamma
from matplotlib import pyplot as plt
#------------------------------------------------------------
# plot the distributions
k_values = [1, 2, 3, 4]
theta_values = [1, 1, 2, 2]
linestyles = ['-', '--', ':', '-.']
x = np.linspace(1E-6, 10, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(5, 3.75))

for k, t, ls in zip(k_values, theta_values, linestyles):
    dist = gamma(k, 0, t)
    plt.plot(x, dist.pdf(x), ls=ls, c='black',
             label=r'$k=%.1f,\ \theta=%.1f$' % (k, t))

plt.xlim(0, 10)
plt.ylim(0, 0.45)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|k,\theta)$')
plt.title('Gamma Distribution')

plt.legend(loc=0)
plt.show()


# In[20]:


from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0,10,.1)
y1 = np.random.gamma(shape=4, scale=3, size=1000) + 2  # sets loc = 2 
y2 = np.hstack((y1, 10*np.random.rand(100)))  # add noise from 0 to 10

# fit the distributions, get the PDF distribution using the parameters
shape1, loc1, scale1 = gamma.fit(y1)
g1 = gamma.pdf(x=x, a=shape1, loc=loc1, scale=scale1)
g1
import seaborn as sns
sns.distplot(g1)


# # Problem 4

# In[11]:


import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[12]:


n = 1000
u = np.random.uniform(low = 0.0, high = 1.0, size = n)
x = u**(1/3)
sns.distplot(x, hist = True, kde=True, bins = 10)
plt.show()


# # Problem 5

# In[13]:


n = 1000
k = 0
j = 0
y = np.zeros(n)
while k<n-1:
    u = np.random.uniform(size = 1)
    j = j+1
    x = np.random.uniform(size = 1)
    if x*(1-x) > u:
        k = k+1
        y[k] = x
p = np.linspace(start = 0.1, stop = 0.9, num=10)


# In[69]:


import pandas as pd
from scipy.stats import beta
from scipy.stats.mstats import mquantiles


# In[75]:


q_hat =  scipy.stats.mstats.mquantiles(y, p)
##q_hat = np.quantile(y,p)
r = beta.ppf(p, 3, 2)
z = np.sqrt(p*(1-p)/(n*beta.pdf(r, 3, 2)**2))


# In[76]:


print(q_hat)
print(r)
print(z)
temp = np.array([q_hat, r])
for i in temp:
    sns.distplot(i, hist = True, kde = True, color = 'darkblue')

plt.show()


# # Problem 6

# In[11]:


import numpy as np
import math
import scipy.stats
import seaborn as sns
import collections as col

n = 1000
theta = 0.5
u = np.random.uniform(size = n)
v = np.random.uniform(size = n)
x = (1+np.log(v)/np.log(1-(1-theta)**u))
x = np.floor(x)

k = []
for i in range(1, int(max(x))+1):
    k.append(i)
    
p = []
for j in range(0, len(k)):
    temp = -1/np.log(1-theta)*(theta*k[j])/k[j]
    p.append(temp)
    
se = []
for i in range(0, len(p)):
    temp = np.sqrt(p[i]*(1-p[i])/n)
    se.append(temp)
    
c = col.Counter(x).values()
c = list(c)
p_hat = []
for i in range(0, len(c)):
    temp = c[i]/n
    p_hat.append(temp)

print("P_hat: ", p_hat)
print("\nP: ", p)
print("\nse: ", se)
sns.distplot(x, hist = True, kde = True, bins = 8)


# # Problem 3

# In[6]:


######################################Problem 3################################
import numpy as np
from scipy.stats import binom
import seaborn as sns
def rlaplace(n, mu, sigma):
    U = np.random.uniform(0,1,n)
    sign = binom.rvs(1, 0.5, size = n)
    sign[sign>0.5] = 1
    sign[sign<0.5] = -1
    y = mu + sign*sigma/np.sqrt(2)*np.log(1-U)
    print(y)
    sns.distplot(y)


# In[10]:


rlaplace(1000,0.5,0.8)

