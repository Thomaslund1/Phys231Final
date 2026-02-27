"""
Created on Tue Feb 20 15:19:15 2024

@author: benjmainansbacher
"""

import csv
import numpy as np
from scipy.integrate import odeint
from scipy.stats import rankdata 
import scipy as scipy
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
import os

#%% ODE 
tStart = 0
tEnd = 10000
tInt = 100000
t = np.linspace(tStart, tEnd , tInt)

#Duffing Eq.
def duffing(k, omega, gamma, alpha, beta, C = 0.15): 
    
    x = []
    def dx_dt(x, t):
        return [x[1], -beta*x[0] - k * x[1] - alpha*x[0]**3 + gamma * np.cos(omega*t) + C]

    #Solve ODE
    xs = odeint(dx_dt, [1,0], t)
    return xs


# Finds time intervals between zero-crossings in a time series
def zero_crossings_time_diff(xs):
    #Zero Crossings
    zero_crossings = np.where(np.diff(np.signbit(xs[:, 1])))[0].tolist()

    # Time @ Zero Crossings
    tAtCross = []
    for i in range(len(zero_crossings)):
        tAtCross.append(t[zero_crossings[i]])

    # Time Diff for Word Creation
    timeDiff = []
    for j in range(len(tAtCross)-1):
        timeDiff.append(np.abs(tAtCross[j]-tAtCross[j+1]))
    return timeDiff


def update_graph(frames): # Replace the variable you care about with W[frames]
    eq = duffing(k=W[frames], 
                 omega=1.2, 
                 gamma=1.0, 
                 alpha=1.0, 
                 beta=-1.0)
    
    x = eq[:, 0]
    x_dot = eq[:, 1]
    
    ax[0].cla()
    
    ax[0].plot(x, x_dot, linewidth=0.7)
    ax[0].set_title(np.round(W[frames], 4))
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('p')
    ax[0].set_xlim(-2, 2) #Change the bounds to match your graph bounds
    ax[0].set_ylim(-2, 2)
    
    ax[1].cla()
    
    newdata = zero_crossings_time_diff(duffing(k=W[frames], omega=1.2, gamma=1.0, alpha=1.0, beta=-1.0))
    symbols1, probs = ansbacher_ordinal_distribution(newdata, dx=3, return_missing=True, tie_precision=8)[:2]
    pops = probs
    ax[1].bar(['012','021','102','120','201','210'], pops, width=0.3)
    ax[1].set_ylim(0,0.5)
    ax[1].set_title('Word Populations')
    ax[1].set_xlabel('Word')
    ax[1].set_ylabel('Frequency')

    
    plt.close(fig)


def ansbacher_ordinal_distribution(data, dx=3, dy=1, taux=1, tauy=1, return_missing=False, tie_precision=None):

# Updated version of ordinal distribution that considers periodic words in the total population caluclation
# but doesn't treat them as individual words in the return    

    try:
        ny, nx = np.shape(data) 
        data   = np.array(data)
    except:
        nx     = np.shape(data)[0]
        ny     = 1
        data   = np.array([data])

    if tie_precision is not None:
        data = np.round(data, tie_precision)

    partitions = np.concatenate(
        [
            [np.concatenate(data[j:j+dy*tauy:tauy,i:i+dx*taux:taux]) for i in range(nx-(dx-1)*taux)] 
            for j in range(ny-(dy-1)*tauy)
        ]
    )

    symbols = np.apply_along_axis(rankdata, 1, partitions, method='min') - 1
    symbols, symbols_count = np.unique(symbols, return_counts=True, axis=0)

    probabilities = symbols_count/len(partitions)

    if return_missing==False:
        return symbols, probabilities
    
    else:
        all_symbols   = list(map(list,list(itertools.permutations(np.arange(dx*dy)))))
        all_probs     = [0]*6
        for i in range(len(all_symbols)):
            for j in range(len(symbols)):
                if np.array_equal(all_symbols[i], symbols[j]):
                    all_probs[i] += probabilities[j]
    
    return all_symbols, all_probs, partitions, symbols, symbols_count

sol = duffing(k=0.3, omega=3.1, gamma=1.0, alpha=1.0, beta=-1.0)
x = sol[550:, 0]
p = sol[550:, 1]

plt.plot(x, p)
plt.title('duffing(k=0.3, omega=2.8, gamma=1.0, alpha=1.0, beta=-1.0')
plt.xlabel('X')
plt.ylabel('P')


symbols, probs, partitions, osymbols, symbols_count = ansbacher_ordinal_distribution(zero_crossings_time_diff(duffing(0,0,0,0,1)), return_missing=True, dx=3, tie_precision=8)
print(symbols, probs, partitions, osymbols, symbols_count)
#print(-sum(probs*np.log(probs))/np.log(6))

p012 = []
p021 = []
p102 = []
p120 = []
p201 = []
p210 = []

entropy=[]

# 3 Dimensional Words
# Calculates PE (entropy) and symmetry groups given a set of data from IPI or zero crossings of the duffing equation
w = np.linspace(2.8, 3.14, num=200)
for w_ in w:
    data = zero_crossings_time_diff(duffing(k=0.3, omega=w_, gamma=1.0, alpha=1.0, beta=-1.0))
    symbols1, probs = ansbacher_ordinal_distribution(data, dx=3, return_missing=True, tie_precision=8)[:2]
    
    p012 += [probs[0]]
    p021 += [probs[1]]
    p102 += [probs[2]]
    p120 += [probs[3]]
    p201 += [probs[4]]
    p210 += [probs[5]]
    
    summation = 0
    for i in range(6):
        logprobs = 0
        if probs[i] == 0:
            logprobs = 0
        else:
            logprobs = np.log(probs[i])
        summation += (-1*(probs[i]*logprobs)/np.log(6))
    entropy += [summation]
    
# Plotting Individual Word Populations
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure(layout='constrained')
ax = fig.add_subplot()

w_range = w

ax.plot(w_range, p012, 'r', linewidth=1, label='012')
ax.plot(w_range, p021, 'b', linewidth=1, label='021')
ax.plot(w_range, p102, 'g', linewidth=1, label='102')
ax.plot(w_range, p120, 'c', linewidth=1, label='120')
ax.plot(w_range, p201, 'm', linewidth=1, label='201')
ax.plot(w_range, p210, 'y', linewidth=1, label='210')
ax.plot(w_range, entropy,'k', linewidth=1, label='entropy')
leg = ax.legend();
ax.set_title(r'Word Populations & PE vs. $\omega$')
ax.set_xlabel('$\omega$')
ax.set_ylabel('Population Size/Entropy')

# Animation 

plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.autolayout"] = True
num_frames = 5


W = np.linspace(0.5, 0.65, 100) # Put the range for your variable


fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(15,8))
    
ani = FuncAnimation(
    fig, 
    update_graph, 
    frames=100, 
    interval = 110, cache_frame_data=False)

path = os.getcwd() + "/test.gif"

print(path)

ani.save('/home/thomas/Desktop/Chaos/test2.gif', writer = 'pillow') # Change the title so you can find it
# %%
