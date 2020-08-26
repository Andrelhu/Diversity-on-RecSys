# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:29:33 2020

@author: andre
"""
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

# 4. Appendix: Additional functions for analysis.
        
# The following functions worked for data wrangling and preparations. No simulation use.

#Get a gini-coefficient value from a list input.
def G(v):
    bins = np.linspace(0., 100., 11)
    total = float(np.sum(v))
    yvals = []
    for b in bins:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)
    # perfect equality area\n",
    pe_area = np.trapz(bins, x=bins)
    # lorenz area\n",
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    return bins, yvals, gini_val

#Plot a single run output.
def Plot_Run(M):
    M = pd.DataFrame(M)
    df2 = M#[M.columns[:-2]]
    
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
 
def plot_many(size,times,classes):
    for k in range(classes):
        temp = []
        for i in range(times):
            temp.append(evaluation(0,[float(rd.randint(-100,100))/100 for i in range(size)],[float(rd.randint(-100,100))/100 for i in range(size)]))
        plt.hist(temp,bins=50)
        plt.show()
   
#Parse data and save to pickle for posterior analysis.   
def manual_clean(cogni='C3',prcnt=50):
    sizes = [[2000,400],[5000,1000],[10000,2000]]
    dff = pd.DataFrame()
    for s in sizes:
        df = pd.read_pickle("["+str(s[0])+", "+str(s[1])+", 'Sociological', 100]")
        dft = load_basic2(df[df.index == 0],s[0],s[1],cogni)
        for j in range(1,len(df)):
            dft = dft.append(load_basic2(df[df.index == j],s[0],s[1],cogni))
        dff = dff.append(dft)
    dff.to_pickle('Results_FSociological_S100_C'+str(cogni)+'_P'+str(prcnt))
    
#Used to load basic features into a pandas dataframe.
def load_basic2(df00,u,p,cogni):
    df0 = get_basics(df00)
    df0['u_size'] = [u]*len(df0)
    df0['p_size'] = [p]*len(df0)
    df0['filter'] = ['Sociological']*len(df0)
    df0['u_cogni'] = [cogni]*len(df0)
    df0['steps'] = ['100']*len(df0)
    return df0

#Supports load_basics by giving aggregating all simulation results
def get_basics(df):
    df2 = df#[M.columns[:-2]]\n"
    viewstats = {'mean':[],'median':[],'std':[],'total':[],'minmax':[]}
    ratingstats = {'mean':[],'median':[],'std':[],'total':[],'minmax':[]}

    for i in df2.values:
            raw_views = []
            raw_rating_mean = []
            for j in i:
                raw_views.append(float(len(j)))
                raw_rating_mean.append(np.mean(j))       #Mind we take totals for views and mean for ratings\n",
            #Get mean, median, stdev, q1, q3, \n",
            viewstats = update_stats(viewstats,raw_views)
            ratingstats = update_stats(ratingstats,raw_rating_mean)
            c,bins,gini = G(np.array(raw_views))
            viewstats['gini'] = gini
            c,bins,gini = G(np.array(raw_rating_mean))
            ratingstats['gini'] = gini

            dfv = pd.DataFrame()
            for k in ['mean','median','std','total','minmax','gini']:
                dfv['v_'+str(k)] = viewstats[str(k)]
            for k in ['mean','median','std','total','minmax','gini']:
                dfv['r_'+str(k)] = ratingstats[str(k)]

            return dfv
        
def update_stats(dct,raw_list):
    dct['mean'].append(np.mean(raw_list))
    dct['median'].append(np.median(raw_list))
    dct['std'].append(np.std(raw_list))
    dct['total'].append(len(raw_list))
    dct['minmax'].append([min(raw_list),max(raw_list)])
    return dct