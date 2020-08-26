# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:51:01 2020

@author: ELHuillier
"""
#Used libraries for simulation and data analysis.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import re
import seaborn as sns
import time
from scipy import spatial
import scipy.stats
from numpy import dot
from numpy.linalg import norm
from collections import Counter
from multiprocessing import Pool
import multiprocessing

#For RecSys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
        
############################      
# 1.a Simulation Agents

class market(object): # 1.a.1 - the Market agent - contains other agents and keeps records of the simulation run. Most importantly, the agent carries the Information Filter mechanism.
    #Initiatialization of Market agent which works as an enviroment for Users, Products, and the Information Filter.
    def __init__(s,A_population,P_population,IFtype='None',run='NONE'):
        s.A,s.P = A_population,P_population
        s.ratings_q = 0
        
        #Create a list of agents. Then create a dataframe that contains the object and all parameters in other columns
        s.agente = [Agent(id_) for id_ in range(s.A)]
        
        #Do the same for Products.
        s.productum = [Product(id_) for id_ in range(s.P)]
        
        #Information Filters
        s.IFtype = IFtype
        print('SimID: '+str(run)+' | P = '+str(s.A)+' | L = '+str(s.P)+' | IF Type: '+str(s.IFtype))
        
        #Hande internal parameters for the simulation run's RecSys method. Depending on filter type.
        s.IFref = {'None':s.search,                  #Used as search only (random search and limited testing)
                   'Cognitive':s.cognitive,          #Search (limit=3) among top recommendations by content-based
                   'Sociological':s.sociological,    #Search (limit=3) among top recs by collaborative-filtering
                   'Sociological_partial':s.sociological,
                   'Collaborative_User_PearsonR':s.pearson_collab,
                   'Collaborative_Item_PearsonR':s.pearson_collab,
                   'Peers':s.peers}        
        s.method = s.IFref[IFtype]
        
        #User/Item Dataset (Users in columns)
        if s.IFtype == 'None':
            s.dataset = pd.DataFrame()
            s.recs = {}
            for userid in range(s.A):
                s.dataset[userid] = [0.0]*s.P
                s.recs[userid] = []
                
        if s.IFtype == 'Collaborative_User_PearsonR':
            s.dataset = pd.DataFrame()
            s.recs = {}
            for userid in range(s.A):
                s.dataset[userid] = [0.0]*s.P
                s.recs[userid] = []
        
        if s.IFtype == 'Collaborative_Item_PearsonR':
            s.dataset = pd.DataFrame()
            s.recs = {}
            for itemid in range(s.P):
                s.dataset[itemid] = [0.0]*s.A
                s.recs[itemid] = []
            
        #Initiate information filter mechanisms (e.g. cosine_similarity for Cognitive filtering)
        if IFtype == 'Cognitive':
            partial = int(len(s.productum[0].features)/2) #only half of the features
            prod_feats = {prod.id:prod.features[:partial] for prod in s.productum} 
            cs_predf = pd.DataFrame.from_dict(prod_feats,orient='index')
            s.cs_df = pd.DataFrame(cosine_similarity(cs_predf))
            s.cs = {}
            for p in s.productum:
                s.cs_df = s.cs_df.sort_values(by=p.id,ascending=False)
                s.cs[p.id] = list(s.cs_df[p.id].index)[1:]
            s.recommended = {}
            s.recsq,s.successrecs = 0,0
            print('Cosine Similarity matrix done!')
        
        elif IFtype == 'Sociological':
            partial = int(len(s.agente[0].preference)  )       
            agente_prefs = {agent.id:agent.preference[:partial] for agent in s.agente} 
            cs_predf = pd.DataFrame.from_dict(agente_prefs,orient='index')
            s.cs_df = pd.DataFrame(cosine_similarity(cs_predf))
            s.cs = {}
            for p in s.agente:
                s.cs_df = s.cs_df.sort_values(by=p.id,ascending=False)
                s.cs[p.id] = list(s.cs_df[p.id].index)[1:]
            s.recommended = {}
            s.recsq,s.successrecs = 0,0
            print('Cosine Similarity matrix done!')     
        
    def step(s,i):
        #Every step the market activates some agents (50% approx), they evaluate a single product and report the feedback.
        #Special process for RecSys with Sociological_partial
        if i in [50,55,60,65,70,75,80,85,90,95,100]:
            if s.IFtype == 'Collaborative_User_PearsonR':
                s.pearson_collab_update() 
            if s.IFtype == 'Collaborative_Item_PearsonR':
                s.pearson_item_collab_update()
        
        #Main procedure of steps function: Go over every agent and activate
        for a in s.agente:            
            if a.activation < rd.random():  #Activation (50% probability of activating and act by searching/rating)          
                #The particular RecSys only operates from the 50th step.
                if i < 50:
                    s.search(a)
                else:
                    s.method(a)
                    
    #Other main functions of the Market agent.    
    def search(s,a): #Refers to User agent search() function. 
        a.search(s)
    
    def cognitive(s,a): #Cognitive process to deliver recommendations through the simulation.
        rec_list = []
        if len(a.consumed) > 0:
            for id_ in a.experience.keys():
                if len(rec_list) >= 3:
                    break                     
                best_recs = s.cs[id_][:10]     
                for recs in best_recs:
                    if recs not in a.consumed:
                        rec_list.append(recs)
        s.recsq += 1   
        if len(rec_list) > 0:
            s.successrecs += 1
            a.search(s,pop=rec_list)
        else:
            a.search(s)
        
    def sociological(s,a): #Sociological process to deliver recommendations through the simulation.
        rec_list = {k:0 for k in range(len(s.productum))}
        for id_ in s.cs[a.id][:50]: #picks top 50
                if len(s.agente[id_].consumed) > 0:
                    t_recs = 0
                    for k,v in s.agente[id_].experience.items():
                        if t_recs == 5: #max 5 best movies per agent
                            break          
                        if v > 0.5:
                            t_recs += 1
                            rec_list[k] += 1
        rec_list = {k: v for k, v in sorted(rec_list.items(), key=lambda item: item[1],reverse=True)}          
        s.recsq += 1   
        if len(rec_list.keys()) > 0:
            s.successrecs += 1
            a.search(s,pop=list(rec_list.keys())[:10])
        else:
            a.search(s)
     
    def pearson_collab_update(s): #Sociological or collaborative filtering with partial updates (work in progress)
        s.recommended = s.dataset.corr(method='pearson')    
        for a in s.agente:
            recs_a = s.recommended[a.id]
            recs_a = recs_a.sort_values(ascending=False)
            recs = []
            for user in list(recs_a.index[1:5]):
                tdic = Counter(s.agente[user].experience)
                recs = recs + [tdic.most_common(4)[j][0] for j in range(0,4)]
            recs = list(set(recs))
            recs = recs[:10]
            s.recs[a.id] = recs
            print(s.recs[a.id])
    
    def pearson_item_collab_update(s):
        s.recommended = s.dataset.corr(method='pearson')
        for a in s.agente:
            tdic = Counter(a.experience)
            top_items = [tdic.most_common(3)[j][0] for j in range(0,3)]
            top_10_related = []
            for i in top_items:
                recs_a = s.recommended[i]
                recs_a = recs_a.sort_values(ascending=False)
                i_top = recs_a.index
                included = 0
                for ii in i_top:
                    if included == 3:
                        break
                    if ii not in a.consumed: 
                        top_10_related.append(ii)
                        included += 1
            s.recs[a.id] = top_10_related
            
    def pearson_collab(s,a):
        if len(s.recs[a.id]) > 2:
            a.search(s,pop=s.recs[a.id])
        else:
            a.search(s)
        
    def cf_partial(s):
        consumed_d = {}
        for agent in s.agente:
            consumed = []
            for p in s.productum:
                if p.id in agent.consumed:
                    consumed.append(1)
                else:
                    consumed.append(0)
            consumed_d[agent.id] = consumed
        cs_consumed = pd.DataFrame.from_dict(consumed_d,orient='index')
        s.cs_df = pd.DataFrame(cosine_similarity(cs_consumed))
        s.cs = {}
        for p in s.agente: #get a df with recommendations for current similarity
            s.cs_df = s.cs_df.sort_values(by=p.id,ascending=False)
            s.cs[p.id] = list(s.cs_df[p.id].index)[1:]
        s.recommended = {}
        s.recsq,s.successrecs = 0,0
        print('Cosine Similarity matrix done for CF partial!')
    
    def top10(s,a): #Ranking recommendation procedure. Take top 10 and recommend those.
        if len(s.top10) > 1:
            rd.shuffle(s.top10)
            for target_top in s.top10:
                if target_top not in a.consumed:
                    a.consume(s,target_top,s.productum[target_top].features)
        else:
            target_p = a.search(s.P)
            if target_p not in a.consumed:
                a.consume(s,target_p,s.productum[target_p].features)     
                
    #Update functions for user and product agents in the Market environment.
    def update_agents(s):
        for id_ in range(s.A):
            s.agente[id_].experience = s.A_df[s.A_df.id == id_].experience
            s.agente[id_].consumed = s.A_df[s.A_df.id == id_].consumed
            
    def update_objs(df,id_,obj_):
        df[df.Object == id_] = obj_
        return df
    
    def update_dataset(s,item_id,user_id,rating):
        if s.IFtype == 'Collaborative_Item_PearsonR':
            s.dataset.at[user_id,item_id] = float(rating)
        else:
            s.dataset.at[item_id,user_id] = rating    
        s.productum[item_id].ratings.append(rating)        # utility=rating is stored in the Market agent/environment
    
        
class Agent(object): # 1.a.2 - the User agent - represent the behavior of a bounded rational individual searching for content consumption.
    def __init__(s,id_): 
        s.id = id_
        s.activation = 0.5
        s.total_utility = 0
        s.preference = [rd.randint(0,2) for i in range(100)]
        s.experience = {}
        s.consumed = []    
            
    def search(s,M,pop=None,limit=3): #The user agent always performs a search (with or without the Info Filter).
        if pop == None:  #If pop (which is expected to be a list of products) is not given then the user will take a maximum number of products randomly. 
            pop = range(0,M.P)
        if limit > len(pop):
            limit = len(pop)        
        targets = rd.sample(pop,limit)  #Take a random sample from the pop by recommendation or random search
        tests = []
        for target_p in targets:
            if target_p not in s.consumed:
                tests.append(s.test(M.productum[target_p].features))
                        
        if len(tests) > 0:  #After testing try to consume
            consume_p = targets[max(range(len(tests)), key=tests.__getitem__)]
            s.consume(consume_p,M.productum[consume_p].features,M)

    def consume(s,movie_id,movie_value,M): #Take the best movie and consume it with 'evaluate()' function. The utility is registered as the rating.
        s.utility = s.evaluate(movie_value)
        s.experience[movie_id] = s.utility
        s.experience = {k: v for k, v in sorted(s.experience.items(), key=lambda item: item[1],reverse=True)}
        s.consumed.append(movie_id)
        M.update_dataset(movie_id,s.id,s.utility)        
    
    def evaluate(s,movie): #This returns a utility from the vector distance between their preferences and the product features
        return 1 - spatial.distance.cosine(s.preference,movie)
    
    def test(s,movie):     #Users may test with limited information. From the limited amount of products (by search limit) they can only test some features of the product without full consumption
        return dot(s.preference[:20], movie[:20])/(norm(s.preference[:20])*norm(movie[:20]))
    
    
class Product(object): # 1.a.3 - the Product object - still very simple, only registers and interacts with Users
    def __init__(s,id_):
        s.id = id_
        s.features = [rd.randint(0,2) for i in range(100)]
        s.ratings = []
        s.views = 0
        
############################  
# 2. Support functions for the simulation.
def Run(P,L,simtype,procnum,return_dict,run): #Basic execution of one simulation
    t0 = time.time()
    M = market(P,L,IFtype=simtype,run=run)
    t1 = time.time()
    for i in range(100):
        if M.IFtype == 'top10' and i > 5:
            M.P_df = M.P_df.sort_values(by='rating',ascending=False)
            M.top10 = list(M.P_df.id)[:10]
        M.step(i)
    t2 = time.time()
    print('Initialization time: '+str(t1-t0)+' secs.\nTotal time: '+str(t2-t1)+' secs.')
    if simtype == 'Cognitive':
        print('Out of '+str(M.recsq)+', '+str(M.successrecs)+' where successful. '+str(float(M.successrecs)/M.recsq))
    return_dict[procnum] = [mov.ratings for mov in M.productum] #get ratings raw values
    return return_dict
      
def Runtest(P,L,simtype,procnum,return_dict,run):
    t0 = time.time()
    M = market(P,L,IFtype=simtype,run=run)
    t1 = time.time()
    for i in range(100):
        if M.IFtype == 'top10' and i > 5:
            M.P_df = M.P_df.sort_values(by='rating',ascending=False)
            M.top10 = list(M.P_df.id)[:10]
        M.step(i)
    t2 = time.time()
    print('Initialization time: '+str(t1-t0)+' secs.\nTotal time: '+str(t2-t1)+' secs.')
    if simtype == 'Cognitive':
        print('Out of '+str(M.recsq)+', '+str(M.successrecs)+' where successful. '+str(float(M.successrecs)/M.recsq))
    return_dict[procnum] = [mov.ratings for mov in M.productum] #get ratings raw values
    return return_dict,M      

def evaluation(s,individual_pref,product_features):        
        normalized_feat = product_features # / np.linalg.norm(product_features)
        normalized_pref = individual_pref # / np.linalg.norm(individual_pref)
        return float(np.dot(normalized_feat,normalized_pref))/len(individual_pref)

############################
# 3. Running the simulation
# Set the values for the different scenarios to be simulated. Each case has [user population, product space size, filter type, number of simulations]
sim_settings = [[2000,400,'Collaborative_User_PearsonR',50]]#,
                #[5000,1000,'None',50],
                #[10000,2000,'None',50]]#,
                #[20000,4000,'Cognitive',100]]#,
                #[40000,8000,'Cognitive',100]]   

#Multiprocessing with Python (set up number of parallel nodes)
if __name__ == '__main__':
    #Md,M = Runtest(200,50,'None',1,{},1)          
    if 1 == 1:                                    
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
        output.to_pickle(str('D:\Simulations\\')+str(setup))