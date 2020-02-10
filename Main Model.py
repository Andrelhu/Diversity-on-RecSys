# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:51:01 2020

@author: ELHuillier
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import re
import seaborn as sns
import time
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
from collections import Counter
from multiprocessing import Pool
import multiprocessing

def evaluation(s,individual_pref,product_features):        
        normalized_feat = product_features # / np.linalg.norm(product_features)
        normalized_pref = individual_pref # / np.linalg.norm(individual_pref)
        return float(np.dot(normalized_feat,normalized_pref))/len(individual_pref)

def plot_many(size,times,classes):
    for k in range(classes):
        temp = []
        for i in range(times):
            temp.append(evaluation(0,[float(rd.randint(-100,100))/100 for i in range(size)],[float(rd.randint(-100,100))/100 for i in range(size)]))
        plt.hist(temp,bins=50)
        plt.show()
        
        
        
########################            ABModel Prototype
class market(object):
    def __init__(s,A_population,P_population,IFtype='None'):
        s.A,s.P = A_population,P_population
        s.ratings_q = 0
        
        #Create a list of agents. Then create a dataframe that contains the object and all parameters in other columns
        s.agente = [Agent(id_) for id_ in range(s.A)]
        #s.agente_dict = {agent.id:[agent.id,agent.total_utility,agent.consumed,agent.experience] for agent in s.agente}
        #s.A_df = pd.DataFrame.from_dict(s.agente_dict,orient='index')
        #s.A_df.columns = ['id','total_utility','consumed','experience']
        
        #prototype fix allocation (keep it numpy for CUDA use)
        #s.A_df.id = [int(i) for i in s.A_df.id]
        #s.A_df.total_utility = [int(i) for i in s.A_df.total_utility]
        #s.A_df.consumed = [[] for i in s.A_df.id]
        
        #Do the same for Products.
        s.productum = [Product(id_) for id_ in range(s.P)]
        #s.product_dict = {product.id:[product.id,product.features,product.rating,product.views] for product in s.productum}
        #s.P_df = pd.DataFrame.from_dict(s.product_dict,orient='index')
        #s.P_df.columns = ['id','value','rating','views']
        
        #Information Filters
        s.IFtype = IFtype
        print('IF Type: '+str(s.IFtype))
        
        s.IFref = {'None':s.search,                  #Used as search only (random search and limited testing)
                   'Cognitive':s.cognitive,          #Search (limit=3) among top recommendations by content-based
                   'Sociological':s.sociological,    #Search (limit=3) among top recs by collaborative-filtering
                   'Peers':s.peers}
        
        s.method = s.IFref[IFtype]
        
    def step(s):
        #Every step the market activates ALL agents, they evaluate a single product and report the feedback.
        #For now, products utility is their value. MEaning that final aggregated 'evaluations' should be a gauss distro.
        for a in s.agente:            
            if a.activation < rd.random():  #Activation (50% prob.)
                s.method(a)

                
    def search(s,a):
        a.search(s)
        
    def cognitive(s,a):
        pass
    
    def sociological(s,a):
        pass
    
    def peers

    def top10(s,a):
        if len(s.top10) > 1:
            rd.shuffle(s.top10)
            for target_top in s.top10:
                if target_top not in a.consumed:
                    a.consume(s,target_top,s.productum[target_top].features)
        else:
            target_p = a.search(s.P)
            if target_p not in a.consumed:
                a.consume(s,target_p,s.productum[target_p].features)     
                
    def update_agents(s):
        for id_ in range(s.A):
            s.agente[id_].experience = s.A_df[s.A_df.id == id_].experience
            s.agente[id_].consumed = s.A_df[s.A_df.id == id_].consumed
            
    def update_objs(df,id_,obj_):
        df[df.Object == id_] = obj_
        return df
    
class Agent(object):
    def __init__(s,id_):
        s.id = id_
        s.activation = 0.5
        s.total_utility = 0
        l = [0,0,1]
        rd.shuffle(l)
        s.preference = [rd.randint(0,2) for i in range(100)]
        #after basic
        s.experience = {}
        s.consumed = []
        
    def search(s,M,pop=None,limit=3):
        #Random
        if pop == None:
            pop = range(0,M.P)
        targets = rd.sample(pop,limit)
        print(targets)
        tests = []
        for target_p in targets:
            if target_p not in s.consumed:
                tests.append(s.test(M.productum[target_p]))
        consume_p = targets[max(xrange(len(tests)), key=values.__getitem__)]
        print(tests)
        print(consume_p)
        print('\n')
        s.consume(consume_p,M.productum[consume_p].features,M)
        
    def consume(s,movie_id,movie_value,M):
        #if s.preference == movie_value:
        #    s.utility = 1
        #else:
        #    s.utility = 0
        s.utility = s.evaluate(movie_value)
        s.experience[movie_id] = s.utility
        s.consumed.append(movie_id)
        M.productum[target_p].ratings.append(s.utility)
        
      #  market.ratings_q += 1
      #  market.A_df.loc[s.id,'experience'][movie_id] = s.utility
      #  market.A_df.loc[s.id,'total_utility'] += s.utility
      #  market.A_df.loc[s.id,'consumed'].append(movie_id)
      #  market.P_df.loc[movie_id,'rating'] += s.utility
      #  market.P_df.loc[movie_id,'views'] += 1
    
    def evaluate(s,movie):
        return 1 - spatial.distance.cosine(s.preference,movie)
    
    def test(s,movie):
        return dot(s.preference, movie)/(norm(s.preference)*norm(movie))
    
    
class Product(object):
    def __init__(s,id_):
        s.id = id_
        s.features = [rd.randint(0,2) for i in range(100)]
        s.ratings = []
        s.views = 0
        
############################ Simulation

sim_obj = []
       
def Run(simtype,procnum,return_dict):
    t0 = time.time()
    #takes 4 hours for 1 run
    M = market(100,20,IFtype=simtype)
  #  M = market(270,40,IFtype=simtype)
    t1 = time.time()
    for i in range(50):
        if M.IFtype == 'top10' and i > 5:
            M.P_df = M.P_df.sort_values(by='rating',ascending=False)
            M.top10 = list(M.P_df.id)[:10]
        M.step()
    t2 = time.time()
    print('Initialization time: '+str(t1-t0)+' secs.\nTotal time: '+str(t2-t1)+' secs.')
    print('\nRatings : '+str(M.ratings_q))
    #M.P_df = M.P_df.sort_values(by='views',ascending=False)
    #return_dict[procnum] = [M.P_df]
    return M

def Plot_Run(MP_df):
    plt.plot(list(MP_df.rating),linewidth=2,alpha=0.5,c='r',label='Rating')
    plt.legend()
    plt.show() 
    plt.plot(list(MP_df.views),linewidth=2,alpha=0.5,c='b',label='Views')
    plt.legend()
    plt.show()

    
if __name__ == '__main__':
    M = Run('None',1,{})
    if 0 == 1:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        #p = Pool(processes=4)
        t=time.time()
        processes = []
        for i in range(0,6):
            p = multiprocessing.Process(target=Run,args=('None',i,return_dict))
            processes.append(p)
            p.start()
            
        for process in processes:
            process.join()
            
        #print(p.map(Run('None'), [1,2,3,4]))
        print(time.time()-t)
        
        more_dict = {}
        t=time.time()
        for i in range(0,6):
            Run('None',i,more_dict)
        print(time.time()-t)    