###########################################################
###                                                     ###
###  Author: Jasmin Bauer (MPI Magdeburg, ARB group)    ###
###                                                     ###
########################################################### 


# Dependencies:
import pandas as pd
import polars as pl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools

# For plotting
import seaborn as sns
sns.set_theme(style="ticks")
sns.axes_style("darkgrid")
sns.set_theme()

# For storing DataFrames
import pickle

# For jupyter notebook 
from IPython.display import display, HTML

# For optimization
from pygmo import *

from itertools import islice
import collections
# for support plotting function doTupleToList
from ast import literal_eval

## Pickle functions

# Save
def doSave(df, path):
    with open(path, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load
def doLoad(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
        return b

###  Indexing
X_0,S_0,P_0 = (i for i in range(3))               
X  ,S  ,P = (i for i in range(3))             
r_S, r_P, mu = (i for i in range(3))  

#########################################################
###
### Support functions
###

## doTswitches
# n_stages = max number of stages the user wants to analyze (e.g. `2` for 2-stage)
# t_start = start time (e.g. `0`)
# t_end = end time (e.g. `24`)
# min_duration = minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
# density = density of time switches (e.g. `1` -> time switches are only tested at full hours; `2` -> time switches can occur every half hour); default=1
# **OUTPUT**= returns a list including arrays containing for each transition of stages the possible tswitch time points 
def doTswitches(n_stages, t_start, t_end, min_duration, density):
    # Generates 

    t_switches=[]
    for i in range(n_stages-1):
        t_switches.append(np.linspace(( t_start + (min_duration*(         i+1))),                                         \
                                      ( t_end   - (min_duration*(n_stages-1-i))),                                         \
                                      int((((t_end   - (min_duration*(n_stages-1-i))) - (t_start + (min_duration*(i+1))))* density)+1)))
        #t_switches.append(np.arange(( t_start + (min_duration*(         i+1))), \
        #                    ( t_end   - (min_duration*(n_stages-1-i))), \
        #                    density))
    return t_switches

## doTswitchesCombis
# t_switches = Output from doTswitches()
# min_duration = minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
# **OUTPUT**= returns a list of all possible combinations of tswitches
def doTswitchesCombis(t_switches, min_duration):
    switches_list = list(itertools.product(*t_switches))
    for i in reversed(range(len(switches_list))):                       # Go over time combination list in reversed order 
        for j in range(len(switches_list[i])-1):                        # Go within each combination and delete time combinations that make no sense (e.g. (5, 1, 3)) 
            if(switches_list[i][j]+min_duration > switches_list[i][j+1]): # the min_duration is as well incorporated here to account for minimal duration of each stage
                switches_list.pop(i)
    return(switches_list)

## doCreateTimesList
# n_stages = max number of stages the user wants to analyze (e.g. `2` for 2-stage)
# t_start = start time (e.g. `0`)
# t_end = end time (e.g. `24`)
# min_duration = minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
# density = density of time switches (e.g. `1` -> time switches are only tested at full hours; `2` -> time switches can occur every half hour); default=1
# **OUTPUT**= returns a list of all possible combinations of tswitches and includes start and end time in each array
def doCreateTimesList(n_stages, t_start, t_end, min_duration, density):
    switches = doTswitches(n_stages, t_start, t_end, min_duration, density)

    switch_combis= doTswitchesCombis(switches, min_duration)
    for i in range(len(switch_combis)):
        switch_combis[i] = (float(t_start),) + switch_combis[i] + (float(t_end),) 
    return switch_combis


# Needed in doBruteForceNum() to skip unnecessary combinations (see below)
# from https://stackoverflow.com/questions/17837316/how-do-i-skip-a-few-iterations-in-a-for-loop
def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)




## doCountCombs
# combi = combinations as list of tuples (e.g. `[(0, 0), (0, 1), ... , (5, 5)]`) (can be generated by itertools.product)
# n_stages = max number of stages the user wants to analyze (e.g. `2` for 2-stage)
# t_start = start time (e.g. `0`)
# t_end = end time (e.g. `24`)
# min_duration = minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
# density = density of time switches (e.g. `1` -> time switches are only tested at full hours; `2` -> time switches can occur every half hour); default=1
# **OUTPUT**= returns the total number of combinations
def doCountCombs(combi, n_stages, t_start, t_end, min_duration, density=1):
    t_switches = doCreateTimesList(n_stages, t_start, t_end, min_duration, density)
    count=0
    for i in range(len(combi)):
        iteration = iter(range(len(t_switches)))
        for j in iteration:
            count=count+1
    return(count)


#########################################################
###
### Analytic solution-based - Brute Force approach:
###

## stage_ana
# see OptMSP_MainFunctions.ipynb for detailed description
# called via doBruteForceAna
def stage_ana(combi, models, s, times, finished=0):

    finished= finished+1
    # Get substrate, product, biomass and end time values for the given stage 
    res = models[int(combi[0])](s, times[0], times[1])

    ## Drop the first element from combination and time
    # e.g.          (0,1,2) and (0.0, 2.0, 3.0, 24.0)
    # will become   (  1,2) and (     2.0, 3.0, 24.0)
    combi.pop(0)
    times.pop(0)

    if((times[0]-res[(len(res)-1)]) < 1e-9):       # next stage (check if substrate went out before stage duration finished)
        if(len(times)==1):              # if t_end is reached -> stop
            return res+[np.Inf]
        return stage_ana(combi, models, res[0:(len(res)-1)], times, finished=finished)
    else:                               # fermentation finished before last stage        
        return res+[finished]

## doBruteForceAna
# see OptMSP_MainFunctions.ipynb for detailed description
def doBruteForceAna(combis, models, n_stages, t_start, t_end, min_duration, s, density=1):
    t_switches = doCreateTimesList(n_stages, t_start, t_end, min_duration, density)

    count=0
    # Initialize polar df
    df = pl.DataFrame(schema={'Index': pl.Int64, 
                              'Times': pl.Utf8, 
                              'Models': pl.Utf8,
                              'End_T': pl.Float64,
                              'End_X': pl.Float64,
                              'End_S': pl.Float64,
                              'End_P': pl.Float64,
                              'finished': pl.Utf8,
                              'Vol_P': pl.Float64,
                              'Y_SubInput': pl.Float64,
                              'Y_SubUsed': pl.Float64})
    
    for i in range(len(combis)):
        curr_combi= combis[i]
        iteration = iter(range(len(t_switches)))
        for j in iteration:
            count=count+1
            res = stage_ana(list(curr_combi), models, s, list(t_switches[j]))

            ## DataFrame
            combi_str       = str(t_switches[j])
            combi_mod_str   = np.array(curr_combi)
            combi_mod_str   += 1
            combi_mod_str   = str(combi_mod_str.tolist())
            Vol_Pield = res[P]/(res[(len(res)-2)]-t_start)
            
            # res.t[0] and res.y[][0] -> 0 because only one time point (at t_end was saved -> see sp.integrate.solve_ivp(... t_eval))
            df_add = pl.DataFrame(data= {  # 'Name'      : [str(names[curr_combi[0]])+' - '+str(combi_t1)+' - '+str(names[curr_combi[1]])+' - '+str(combi_t2)+' -> '+str(names[curr_combi[2]])],
                                            'Index'     : [count],
                                            'Times'     : [combi_str], 
                                            'Models'    : [combi_mod_str],
                                            'End_T'     : [np.round(res[(len(res)-2)], decimals=2)],  # End time of fermentation
                                            'End_X'     : [np.round(res[X], decimals=2)],  #                           corresponding Biomass value
                                            'End_S'     : [np.round(res[S], decimals=2)],  #                           corresponding Substrate value
                                            'End_P'     : [np.round(res[P], decimals=2)],  #                           corresponding Product value
                                            'finished'  : [str(res[(len(res)-1)])], # Was all substrate taken up?
                                            'Vol_P'     : [np.round(Vol_Pield, decimals=2)], # Volumetric yield
                                            'Y_SubInput': [np.round((res[P]/s[S]), decimals=2)],
                                            'Y_SubUsed' : [np.round((res[P]/(s[S]-res[S])), decimals=2)]
                                            })
            # Append current data to df 
            df = df.extend(df_add)

            if(res[len(res)-1] == 1):        # Get new combinations, because fermentation finished in 1. Stage already
                    break
            
            if(res[len(res)-1] < n_stages):  # Skip unneccessary combinations if fermentation finished before last stage
                jj=j
                try:
                    while(t_switches[j][res[len(res)-1]-1] == t_switches[jj][res[len(res)-1]-1]):
                        if(t_switches[j][res[len(res)-1]-1] != t_switches[jj+1][res[len(res)-1]-1]):
                             break
                        consume(iteration, 1)
                        jj += 1 
                except:
                     break
    return(df.to_pandas())


#########################################################
###
### Simulation-based - Brute Force approach:
###

## stage_num
# see OptMSP_MainFunctions.ipynb for detailed description
#
# res = sp.integrate.solve_ivp() (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) 
# called via doBruteForceNum
def stage_num(combi, models, s, times, finished=0, step=0.01):

    finished= finished+1
    # Get substrate, product, biomass and end time values for the given stage 
    def event_Sub0(t,y):
        if(y[S] <= 0.001):
            return 0    # 0 that event triggers
        return 1

    # Event to track when 85% of theoretical max product was produced 
    def event_Pro85percent(t,y):
        if((y[P]/(S_0*6)) >= 0.85):
            return 0
        return 1

    r = sp.integrate.solve_ivp(models[int(combi[0])], t_span=[times[0], times[1]], y0=s, t_eval=[times[1]], events=[event_Sub0, event_Pro85percent], dense_output=True, max_step=step)

    try:    # Case in which event triggers and fermentation stopps before tend
        res = [ r.y_events[0][0][species] for species in range(len(s)) ]
        res = res + [r.t_events[0][0]]
    except: # Case in which fermentation runs until tend
        res = [ r.y[species][0] for species in range(len(s)) ]
        res = res + [r.t[0]]
    ## Drop the first element from combination and time
    # e.g.          (0,1,2) and (0.0, 2.0, 3.0, 24.0)
    # will become   (  1,2) and (     2.0, 3.0, 24.0)
    combi.pop(0)
    times.pop(0)

    if((times[0]-res[(len(res)-1)]) < 1e-9):       # next stage 
        if(len(times)==1):              # if t_end is reached -> stop
            return res+[np.Inf]
        return stage_num(combi, models, res[0:(len(res)-1)], times, finished=finished)
    else:                               # fermentation finished before last stage        
        return res+[finished]


## doBruteForceNum
# see OptMSP_MainFunctions.ipynb for detailed description
def doBruteForceNum(combis, models, n_stages, t_start, t_end, min_duration, s, step=0.01, density=1):
    t_switches = doCreateTimesList(n_stages, t_start, t_end, min_duration, density)

    count=0
    # Initialize polar df
    df = pl.DataFrame(schema={'Index': pl.Int64, 
                              'Times': pl.Utf8, 
                              'Models': pl.Utf8,
                              'End_T': pl.Float64,
                              'End_X': pl.Float64,
                              'End_S': pl.Float64,
                              'End_P': pl.Float64,
                              'finished': pl.Utf8,
                              'Vol_P': pl.Float64,
                              'Y_SubInput': pl.Float64,
                              'Y_SubUsed': pl.Float64})
    
    for i in range(len(combis)):
        curr_combi= combis[i]
        iteration = iter(range(len(t_switches)))
        for j in iteration:
            count=count+1
            res = stage_num(list(curr_combi), models, s, list(t_switches[j]), step=step)

            ## DataFrame
            combi_str       = str(t_switches[j])
            combi_mod_str   = np.array(curr_combi)
            combi_mod_str   += 1
            combi_mod_str   = str(combi_mod_str.tolist())
            # Volumetric yield
            try:
                Vol_Pield=res.y_events[0][0][P]/(res.t_events[0][0]-t_start)
            except: # if fermentation was not finished take product at t_end and devide by t_end to get vol. yield
                Vol_Pield=res[P]/(res[(len(res)-2)]-t_start)

            df_add = pl.DataFrame(data= {  # 'Name'      : [str(names[curr_combi[0]])+' - '+str(combi_t1)+' - '+str(names[curr_combi[1]])+' - '+str(combi_t2)+' -> '+str(names[curr_combi[2]])],
                                            'Index'     : [count],
                                            'Times'     : [combi_str], 
                                            'Models'    : [combi_mod_str],
                                            'End_T'     : [np.round(res[(len(res)-2)], decimals=2)],  # End time of fermentation
                                            'End_X'     : [np.round(res[X], decimals=2)],  #                           corresponding Biomass value
                                            'End_S'     : [np.round(res[S], decimals=2)],  #                           corresponding Substrate value
                                            'End_P'     : [np.round(res[P], decimals=2)],  #                           corresponding Product value
                                            'finished'  : [str(res[len(res)-1])], # Was all substrate taken up?
                                            'Vol_P'     : [np.round(Vol_Pield, decimals=2)], # Volumetric yield
                                            'Y_SubInput': [np.round((res[P]/s[S]), decimals=2)],
                                            'Y_SubUsed' : [np.round((res[P]/(s[S]-res[S])), decimals=2)]
                                            })
            # Append current data to df 
            df = df.extend(df_add)

            if(res[len(res)-1] == 1):        # Get new combinations, because fermentation finished in 1. Stage already
                    break
            
            if(res[len(res)-1] < n_stages):  # Skip unnecessary combinations if fermentation finished before last stage
                jj=j
                try:
                    while(t_switches[j][res[len(res)-1]-1] == t_switches[jj][res[len(res)-1]-1]):
                        if(t_switches[j][res[len(res)-1]-1] != t_switches[jj+1][res[len(res)-1]-1]):
                             break
                        consume(iteration, 1)
                        jj += 1 
                except:
                     break
    return(df.to_pandas())





#########################################################
###
###     Optimizer approach:
###

## doConvert()
# see OptMSP_MainFunctions.ipynb for detailed description
def doConvert(opt_res, models_num, t_start, t_end, s, step=0.01):
    count=0

    df = pl.DataFrame(schema={'Index': pl.Int64, 
                              'Times': pl.Utf8, 
                              'Models': pl.Utf8,
                              'End_T': pl.Float64,
                              'End_X': pl.Float64,
                              'End_S': pl.Float64,
                              'End_P': pl.Float64,
                              'finished': pl.Utf8,
                              'Vol_P': pl.Float64,
                              'Y_SubInput': pl.Float64,
                              'Y_SubUsed': pl.Float64})
    combi_list = [([t_start]+list(x)+[t_end]) for x in opt_res[0]]
    mod_list = [[ int(y-1) for y in list(x)] for x in opt_res[1]]
    
    for i in range(len(opt_res[0])):
        count=count+1

        res = stage_num(list(mod_list[i]), models_num, s, list(combi_list[i]), step=step)

        # Volumetric yield
        try:
            Vol_P=res.y_events[0][0][P]/(res.t_events[0][0]-t_start)
        except: # if fermentation was not finished take product at t_end and devide by t_end to get vol. yield
            Vol_P=res[P]/(res[(len(res)-2)]-t_start)
            
        ## DataFrame
        df_add = pl.DataFrame(data= {  # 'Name'      : [str(names[curr_combi[0]])+' - '+str(combi_t1)+' - '+str(names[curr_combi[1]])+' - '+str(combi_t2)+' -> '+str(names[curr_combi[2]])],
                                        'Index'     : [count],
                                        'Times'     : [str(list(np.round(combi_list[i], decimals=2)))], 
                                        'Models'    : [str((np.array(mod_list[i])+1).tolist())],
                                        'End_T'     : [np.round(res[(len(res)-2)], decimals=2)],  # End time of fermentation
                                        'End_X'     : [np.round(res[X], decimals=2)],  #                           corresponding Biomass value
                                        'End_S'     : [np.round(res[S], decimals=2)],  #                           corresponding Substrate value
                                        'End_P'     : [np.round(res[P], decimals=2)],  #                           corresponding Product value
                                        'finished'  : [str(res[(len(res)-1)])], # Was all substrate taken up?
                                        'Vol_P'     : [np.round(Vol_P, decimals=2)], # Volumetric yield
                                        'Y_SubInput': [np.round((res[P]/s[S]), decimals=2)],
                                        'Y_SubUsed' : [np.round((res[P]/(s[S]-res[S])), decimals=2)]
                                        })
        # Append current data to df 
        df = df.extend(df_add)

    return(df.to_pandas())


## Logging support function:
# This functions enables to store relevant values of each iteration that can then be transformed into a DataFrame later
# for more information see: https://esa.github.io/pygmo2/tutorials/udp_meta_decorator.html
# self = decorator_problem object 
# dv   = decision vector
def f_log_decor(orig_fitness_function):

    def new_fitness_function(self, dv):
        if hasattr(self, "dv_log"):
            sol = orig_fitness_function(self, dv)
            # Score = sol[0]
            # dv = t_switches and modules

            combi=str(tuple(dv[(self.inner_problem.get_nix()-1):len(dv)]))
            times=str(tuple(dv[0:(self.inner_problem.get_nix()-1)]))
            #self.dv_log.append([sol[0]]+list(dv))
            self.dv_log.append([sol[0]]+[combi]+[times])
            return sol
        else:
            self.dv_log = [dv]
            return orig_fitness_function(self, dv)
        
    return new_fitness_function



## For the optimizer approach the package pygmo (https://esa.github.io/pygmo2/) with the IHS algorithm was used
# The method requires to define an own class providing the customized fitness function, bounds, constraints and support functions
# see OptMSP_MainFunctions.ipynb for detailed description
class Optimizer:
    def __init__(self, s, models, tstart, tend, max_stage, min_duration, objective, extracon=None, step=0.01):
        self.y          = s 
        self.models     = models
        self.tstart     = tstart
        self.tend       = tend
        self.max_stage  = max_stage
        self.min_duration = min_duration
        self.objective  = objective
        self.extracon   = extracon
        self.step       = step
        self.bounds     = None
        self.doBounds()
        #self.extra_info = " "


    def doBounds(self):
        # Initialize lists
        lb = []
        ub = []

        # fill with t_switch bounds
        for i in range(self.max_stage-1): # number of t_switches = number of stages - 1
            lb.append(( self.tstart + (self.min_duration*(i+1))))
            ub.append(( self.tend - (self.min_duration*(self.max_stage-1-i))))

        

        # add bounds of the parameter table indices
        mods_lb = [0]            * self.max_stage
        mods_ub = [(len(self.models)-1)] * self.max_stage

        # Combine to the final lists
        lb = lb + mods_lb
        ub = ub + mods_ub
        self.bounds = (lb, ub)

    def doIneqs(self, x):
        ic = []
        for i in range(self.max_stage-2):
            ic.append((x[i]+self.min_duration)-x[i+1]) # includes the minimal duration for one stage
        
        return ic


    def fitness(self, x):

        ## Structure of x:
        # for e.g. 2 Stage
        # [19.69324898  4.74485318  1.          0.          1.        ]
        # First two are the two switching times and 
        # the last three the indecies for the models list
        ics = self.doIneqs(x=x) # negative values are associated to satisfied inequalities

        # Skip combinations that are not meeting the inequality constraints
        if any(n > 0 for n in ics):
            try:                # case in which extra constraints are present
                return ([0] + ics + [0]*len(self.extracon))
            except(TypeError):  # case in which NO extra constraints are present
                return ([0] + ics)

        ## Optimize
        res = self.objective(self,x)

        # check if extracon is defined and if yes include in inequality constraints
        if self.extracon is not None:
            for i in range(len(self.extracon)):
                ics = ics + [(self.extracon[i]-res[i+1])] # negative values are associated to satisfied inequalities

        # Return score as well as all constraints
        try:
            return([res[0]] + ics) # In pagmo minimization is always assumed; to maximize some objective function, put minus sign in front of objective.
        except(IndexError):
            return([res] + ics)

    def get_bounds(self):
        return self.bounds
    
    # Integer Dimension of the decision vector x
    def get_nix(self):
        return self.max_stage
    
    def get_nec(self):
        return 0

    # Inequality Constraints
    def get_nic(self):
        try:                # case in which extra constraints are present
            return self.max_stage-2+len(self.extracon)
        except(TypeError):  # case in which NO extra constraints are present
            return self.max_stage-2



#############################
###
### Plotting function
###

# Support functions for plotting



# ... for the brute force approach 
def doTupleToList(df):
    # Transforming the string DataFrame columns "Times" and "Models" to numeric values as list in a list (by list comprehension)
    combi_list = [literal_eval(x) for x in df['Times']]
    mod_list = [literal_eval(x) for x in df['Models']]

    return combi_list, mod_list

# ... for the optimizer approach 
def doOptDF(df, n_best):
    df = df.iloc[: , [0, 1, 2]]
    df.columns = ["Score", "Models", "Times"]

    df=df.sort_values(["Score"], ascending=[True]).dropna(axis=0)[:n_best]
    df["Score"] *= -1  # For optimization the objective value had to be made negative, this is now reversed back to normal
    # Incorporate tstart and tend in the times list
    combi_list = [(list(literal_eval(x))) for x in df['Times']]
    # Nested list comprehension for converting string back to list of integers and correct indices to actual module numbers (index+1)
    mod_list = [[ int(y+1) for y in list(literal_eval(x))] for x in df['Models']]
    return combi_list, mod_list, df

## Adjusted stage function for plotting
def stage_num_plotting(combi, models, s, times, step=0.01, results=pd.DataFrame()):
    # r = sp.integrate.solve_ivp(sim, t_span=[times[0], times[1]], y0=s, t_eval=[times[1]], events=[event_Sub0, event_Pro85percent], args=(p[combi[0],0:3], ), dense_output=True, max_step=step)
    r = sp.integrate.solve_ivp(models[combi[0]], t_span=[times[0], times[1]], y0=s, t_eval=[times[1]], dense_output=True, max_step=step)
    t = np.linspace(times[0], times[1], (int(times[1])+1)*20) # (times[1]+1)*20 = Resolution
    try:
        res = [r.y_events[0][0][X], r.y_events[0][0][S], r.y_events[0][0][P], r.t_events[0][0]]
    except:
        res = [r.y[X][0], r.y[S][0], r.y[P][0], r.t[0]]

    dat_add= pd.DataFrame(data={    'Time [h]'  : t, 
                                    'Biomass'   : r.sol(t)[0],
                                    'Substrate' : r.sol(t)[1],
                                    'Product'   : r.sol(t)[2]})
    results= pd.concat([results, dat_add])

    ## Drop the first element from combination and time
    # e.g.          (0,1,2) and (0.0, 2.0, 3.0, 24.0)
    # will become   (  1,2) and (     2.0, 3.0, 24.0)
    combi.pop(0)
    times.pop(0)

    # Calls itself again and again until list of times has length=1 which signals that t_end is reached
    if(len(times)==1):              # if t_end is reached -> stop
        return res+[np.Inf], results
    return stage_num_plotting(combi, models, res[0:3], times, results=results)

## Main plotting function
# df    = dataframe you get from:
#           brute force functions (doBruteForceNum and doBruteForceAna) 
#           optimizer function (e.g. pd.DataFrame(outcome.problem.extract(decorator_problem).dv_log))
# models_num = numeric models
# s     = start values for biomass, substrate, product [numpy array]
# title = title of plot displayed at the top [string]
def customPlot(df, models_num, s, title, step=0.01):
   
    infos = doTupleToList(df)

    times  = infos[0]
    combis = infos[1]
    
    dat = pd.DataFrame()
    palette = sns.color_palette("viridis",n_colors=len(combis)) # https://seaborn.pydata.org/tutorial/color_palettes.html
    fig, axs = plt.subplots(nrows=len(s), ncols=1, sharex=True, figsize=(5,len(s)*5))
    combi_list = []
    
    ## The loop for all the different modi
    for i in range(len(combis)):
        curr_combi= combis[i]
        # Prepare for correct indexing starting from 0
        curr_combi = [x-1 for x in curr_combi]

        ## Simulate ODE
        res = stage_num_plotting(combi=list(curr_combi), models=models_num, s=s, times=list(times[i]), step=step)
        results = res[1]

        # Change from indexing back to numbering models
        curr_combi = [int(x+1) for x in curr_combi]
        times[i] = [round(x,2) for x in times[i]]

        results['Times']=str(curr_combi)+" "+str(times[i])

        ## DataFrame
        combi_str       = " ".join(str(x) for x in (curr_combi,times[i])) # For displaying in the plot

        dat= pd.concat([dat, results]) # res[1] = DataFrame with all Time, Biomass, Substrate and Product columns and times in rows
        combi_list.append(combi_str)


    # Biomass plot
    sns.lineplot(
        data=dat,
        x="Time [h]", y="Biomass",
        hue="Times", #col=columns,
        #kind="line",# size_order=["T1", "T2"], 
        palette=palette,
        #height=5, aspect=.75, facet_kws=dict(sharex=False), 
        linewidth=0.5,
        ax=axs[0],
        legend=True
    ).set(title=title, ylim=(0, dat["Biomass"].max()*1.05))
    plt.setp(axs[0].get_legend().get_texts(), fontsize='8')
    plt.setp(axs[0].get_legend().get_title(), fontsize='10')

    
    # Substrate plot
    sns.lineplot(
        data=dat,
        x="Time [h]", y="Substrate",
        hue="Times", #col="align",
        #kind="line",# size_order=["T1", "T2"], 
        palette=palette,
        #height=5, aspect=.75, facet_kws=dict(sharex=False), 
        linewidth=0.5,
        ax=axs[1],
        legend=True
    ).set(ylim=(0, int(s[S])))
    plt.setp(axs[1].get_legend().get_texts(), fontsize='8')
    plt.setp(axs[1].get_legend().get_title(), fontsize='10')
    #plt.setp(axs[1].set_ylabel('Glucose'))
    

    # Product plot
    sns.lineplot(data=dat, 
        x="Time [h]", y="Product", 
        hue="Times", #col="align",
        #kind="line",# size_order=["T1", "T2"], 
        palette=palette,
        #height=5, aspect=.75, facet_kws=dict(sharex=False), 
        linewidth=0.5,
        ax=axs[2],
        legend=True
    ).set(ylim=(0, dat["Product"].max()*1.05))
    plt.setp(axs[2].get_legend().get_texts(), fontsize='8')
    plt.setp(axs[2].get_legend().get_title(), fontsize='10')
    #plt.setp(axs[2].set_ylabel('Lactate'))
    axs[2].axhline((s[S]*6)*0.85) # 85% theoretical product 
    

    for i in range(len(s)):
        for line, name in zip(axs[i].lines, combi_list):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1]
            if not np.isfinite(y):
                y=next(reversed(line.get_ydata()[~line.get_ydata().mask]),float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
                continue     
            text = axs[i].annotate( name,
                                    xy=(x, y),
                                    xytext=(0, 0),
                                    color=line.get_color(),
                                    xycoords=(axs[i].get_xaxis_transform(),
                                    axs[i].get_yaxis_transform()),
                                    textcoords="offset points",
                                    fontsize=7)
            text_width = (text.get_window_extent(fig.canvas.get_renderer()).transformed(axs[i].transData.inverted()).width)
            if np.isfinite(text_width):
                axs[i].set_xlim(axs[i].get_xlim()[0], text.xy[0] + text_width * 1.05)

