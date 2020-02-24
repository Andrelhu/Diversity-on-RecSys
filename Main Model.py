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
#For rec sys
import nltk
nltk.download('stopwords')
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


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
    def __init__(s,A_population,P_population,IFtype='None',run='NONE'):
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
        print('SimID: '+str(run)+' | P = '+str(s.A)+' | L = '+str(s.P)+' | IF Type: '+str(s.IFtype))
        
        s.IFref = {'None':s.search,                  #Used as search only (random search and limited testing)
                   'Cognitive':s.cognitive,          #Search (limit=3) among top recommendations by content-based
                   'Sociological':s.sociological,    #Search (limit=3) among top recs by collaborative-filtering
                   'Peers':s.peers}
        
        s.method = s.IFref[IFtype]
        
        #Initiate info filter mechanism
        if IFtype == 'Cognitive':
            partial = int(len(s.productum[0].features))
            prod_feats = {prod.id:prod.features[:partial] for prod in s.productum} #only half of the features
            cs_predf = pd.DataFrame.from_dict(prod_feats,orient='index')
            s.cs_df = pd.DataFrame(cosine_similarity(cs_predf))
            s.cs = {}
            for p in s.productum:
                s.cs_df = s.cs_df.sort_values(by=p.id,ascending=False)
                s.cs[p.id] = list(s.cs_df[p.id].index)[1:]
            s.recommended = {}
            s.recs,s.successrecs = 0,0
            print('Cosine Similarity matrix done!')
        
        elif IFtype == 'Sociological':
            partial = int(len(s.agente[0].preference))
            agente_prefs = {agent.id:agent.preference[:partial] for agent in s.agente} #only half of the features
            cs_predf = pd.DataFrame.from_dict(agente_prefs,orient='index')
            s.cs_df = pd.DataFrame(cosine_similarity(cs_predf))
            s.cs = {}
            for p in s.productum:
                s.cs_df = s.cs_df.sort_values(by=p.id,ascending=False)
                s.cs[p.id] = list(s.cs_df[p.id].index)[1:]
            s.recommended = {}
            s.recs,s.successrecs = 0,0
            print('Cosine Similarity matrix done!')
            
        
    def step(s):
        #Every step the market activates ALL agents, they evaluate a single product and report the feedback.
        #For now, products utility is their value. MEaning that final aggregated 'evaluations' should be a gauss distro.
        for a in s.agente:            
            if a.activation < rd.random():  #Activation (50% prob.)
                s.method(a)
            if a.id == 1 or a.id == '1':
                print(a.experience)
                
    def search(s,a):
        a.search(s)
    
    def cognitive(s,a):
        rec_list = []
        if len(a.consumed) > 0:
            for id_ in a.experience.keys():
                if len(rec_list) >= 3:
                    break
                best_recs = s.cs[id_][:10]
                for recs in best_recs:
                    if recs not in a.consumed:
                        rec_list.append(recs)
        s.recs += 1   
        if len(rec_list) > 0:
            s.successrecs += 1
            a.search(s,pop=rec_list)
        else:
            a.search(s)
    
    def sociological(s,a):
        rec_list =[]
        for id_ in s.cs[a.id][:50]: #picks top 50
                if s.agente[id_].consumed > 0:
                    for k,v in s.agente[id_].experience.items():
                        if v > 0.5:
                            pass
                best_recs = s.cs[id_][:10]
                for recs in best_recs:
                    if recs not in a.consumed:
                        rec_list.append(recs)
        s.recs += 1   
        if len(rec_list) > 0:
            s.successrecs += 1
            a.search(s,pop=rec_list)
        else:
            a.search(s)
    
    def peers(s,a):
        pass

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
        s.preference = [rd.randint(0,2) for i in range(100)]
        #after basic
        s.experience = {}
        s.consumed = []
        
    def search(s,M,pop=None,limit=3):
        #Random
        if pop == None:
            pop = range(0,M.P)
        targets = rd.sample(pop,limit)
        tests = []
        for target_p in targets:
            if target_p not in s.consumed:
                tests.append(s.test(M.productum[target_p].features))
        if len(tests) > 0:
            consume_p = targets[max(range(len(tests)), key=tests.__getitem__)]
            s.consume(consume_p,M.productum[consume_p].features,M)

    def consume(s,movie_id,movie_value,M):
        s.utility = s.evaluate(movie_value)
        s.experience[movie_id] = s.utility
        s.experience = {k: v for k, v in sorted(s.experience.items(), key=lambda item: item[1],reverse=True)}
        s.consumed.append(movie_id)
        M.productum[movie_id].ratings.append(s.utility)        
    
    def evaluate(s,movie):
        return 1 - spatial.distance.cosine(s.preference,movie)
    
    def test(s,movie):
        return dot(s.preference[:20], movie[:20])/(norm(s.preference[:20])*norm(movie[:20]))
    
    
class Product(object):
    def __init__(s,id_):
        s.id = id_
        s.features = [rd.randint(0,2) for i in range(100)]
        s.ratings = []
        s.views = 0
        
############################ Simulation

sim_obj = []
       
def Run(P,L,simtype,procnum,return_dict,run):
    t0 = time.time()
    #takes 4 hours for 1 run
    M = market(P,L,IFtype=simtype,run=run)
  #  M = market(270,40,IFtype=simtype)
    t1 = time.time()
    for i in range(50):
        if M.IFtype == 'top10' and i > 5:
            M.P_df = M.P_df.sort_values(by='rating',ascending=False)
            M.top10 = list(M.P_df.id)[:10]
        M.step()
    t2 = time.time()
    print('Initialization time: '+str(t1-t0)+' secs.\nTotal time: '+str(t2-t1)+' secs.')
    #print('\nRatings : '+str(M.ratings_q))
    if simtype == 'Cognitive':
        print('Out of '+str(M.recs)+', '+str(M.successrecs)+' where successful. '+str(float(M.successrecs)/M.recs))
    #get ratings raw values
    return_dict[procnum] = [mov.ratings for mov in M.productum]
    
    #M.P_df = M.P_df.sort_values(by='views',ascending=False)
    return return_dict
    

def Plot_Run(M):
    # = pd.DataFrame(M)
    df2 = M[M.columns[:-2]]
    
    views,rating_mean=[],[]
    for i in df2.values:
        raw_views = []
        raw_rating_mean = []
        for j in i:
            raw_views.append(float(len(j)))
            raw_rating_mean.append(np.mean(j))
        views = views + raw_views
        rating_mean = rating_mean + raw_rating_mean

    df = pd.DataFrame()
    df['views'] = [len(i) for i in df2.values[1]]#np.array(views)
    df['rating'] = [np.mean(i) for i in df2.values[1]]#np.aray(rating_mean)
    gg = sns.jointplot(x='views',y='rating',data=df)
    plt.show()
    
    df = df.sort_values(by='rating',ascending=False)
    plt.plot(list(df.rating),linewidth=2,alpha=0.5,c='r',label='Rating')
    plt.legend()
    plt.show() 
    df = df.sort_values(by='views',ascending=False)
    plt.plot(list(df.views),linewidth=2,alpha=0.5,c='b',label='Views')
    plt.legend()
    plt.show()
 
    
 #values respectively: P, L, IFtype, number of experiments
sim_settings = [[2000,400,'Cognitive',100],
                [5000,1000,'Cognitive',100],
                [10000,2000,'Cognitive',100]]#,
                #[20000,4000,'Cognitive',100],
                #[40000,8000,'Cognitive',100]]   
#sim_settings = [[80000,16000,'Cognitive',100]]#,
             #   [160000,32000,'None',100],
             #   [320000,64000,'None',100]
             #   ]

if __name__ == '__main__':
    M = Run(2000,500,'None',1,{},1)
    if 0 == 1:
     for setup in sim_settings:
        sim_results = {}
        Mp,Ml,Mif,runs = setup[0],setup[1],setup[2],setup[3]
        while len(sim_results) < runs:    
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            #p = Pool(processes=4)
            t=time.time()
            processes = []
            for i in range(0,6):
                p = multiprocessing.Process(target=Run,args=(Mp,Ml,Mif,i,return_dict,len(sim_results)))
                processes.append(p)
                p.start()
                
            for process in processes:
                process.join()
            
            for vals in return_dict.values():
                sim_results[len(sim_results)] = vals

            print(time.time()-t)
        
        output = pd.DataFrame.from_dict(sim_results,orient='index')
        output.to_pickle(str('C:\Simulations\\')+str(setup))
        
        
        