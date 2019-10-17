# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import re
import seaborn as sns
import time
from collections import Counter
from pylab import rcParams
rcParams['figure.figsize'] = 11,8
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=11)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title

def update_obj(df,id_,obj_):
    df[df.Object == id_] = obj_
    return df
    
#ABM Prototype
class market(object):
    def __init__(s,A_population,P_population):
        s.A,s.P = A_population,P_population
        
        #Create a list of agents. Then create a dataframe that contains the object and all parameters in other columns
        s.agente = [Agent(id_) for id_ in range(s.A)]
        s.agente_dict = {agent.id:[agent,agent.experience,agent.consumed] for agent in s.agente}
        s.A_df = pd.DataFrame.from_dict(s.agente_dict,orient='index')
        s.A_df.columns = ['Object','experience','consumed']
        
        #Do the same for Products.
        s.productum = [Product(id_) for id_ in range(s.P)]
        s.product_dict = {product.id:[product,product.value,product.rating,product.views] for product in s.productum}
        s.P_df = pd.DataFrame.from_dict(s.product_dict,orient='index')
        s.P_df.columns = ['Object','value','rating','views']
        
    def step(s):
        #Every step the market activates ALL agents, they evaluate a single product and report the feedback.
        #For now, products utility is their value. MEaning that final aggregated 'evaluations' should be a gauss distro.
        for a in s.agente:
            #Activation
            if a.activation > rd.random():
                target_p = a.search(s.P)
                a.consume(s,target_p,s.productum[target_p].value)
                #Update agents info
                s.A_df = update_obj(s.A_df,a.id,a)
        
class Agent(object):
    def __init__(s,id_):
        s.id = id_
        s.activation = 1
        s.experience = {}
        s.consumed = []
        
    def search(s,pop):
        #Random
        return rd.randint(0,pop-1)
        
    def consume(s,market,movie_id,movie_value):
        s.utility = movie_value
        s.experience[movie_id] = s.utility
        market.A_df.loc[s.id,'experience'] = s.utility
        
        s.consumed.append(movie_id)
        market.A_df.loc[s.id,'consumed'] = s.consumed
        
        market.P_df.loc[movie_id,'rating'] += s.utility
        market.P_df.loc[movie_id,'views'] += 1
    
class Product(object):
    def __init__(s,id_):
        s.id = id_
        s.value = rd.gauss(0,1)
        s.rating = 0
        s.views = 0
        
def go(a=500,p=100):
    t0 = time.process_time()
    M = market(a,p)
    t1 = time.process_time()
    for i in range(100):
        M.step()
    t2 = time.process_time()
    print('Initialization: '+str(t1-t0)+' secs. Total: '+str(t2-t1)+' secs.')
    
    plt.hist(list(M.P_df.rating),bins=10)
    plt.show()
    return M