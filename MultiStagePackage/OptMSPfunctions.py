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
# for support plotting function do_tuple_to_list
from ast import literal_eval


## Pickle functions
def do_save(df, path):
    '''
    This function saves a dataframe as pickle object

    Parameters:
    df (dataframe)  : pandas dataframe 
    path (string)   : path where to save pandas dataframe + name of pickle object (e.g. '/path/to/directory/dataframe_name.pickle')
    
    Returns:
    -
    '''
    with open(path, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def do_load(path):
    '''
    This function loads a pickle object

    Parameters:
    path (string)   : path where the pickle object is located
    
    Returns:
    loaded object (here pandas dataframe from optimization)
    '''
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
        return b


##################################################################################################################
###
###                 Support functions:
###
##################################################################################################################



def do_tswitches(n_stages, t_start, t_end, min_duration, density):
    '''
    Generates all possible switch time points in a list including arrays containing for each transition of stages the possible tswitch time points 
    
    Parameters:
    n_stages (int): max number of stages the user wants to analyze (e.g. `2` for 2-stage)
    t_start (int or float): start time (e.g. `0`)
    t_end (int or float): end time (e.g. `24`)
    min_duration (int or float): minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
    density (int): density of time switches (e.g. `1` -> time switches are only tested at full hours; `2` -> time switches can occur every half hour); default=1
    
    Returns:
    list of arrays 
    '''

    t_switches=[]
    for i in range(n_stages-1):
        t_switches.append(np.linspace(( t_start + (min_duration*(         i+1))),                                         \
                                      ( t_end   - (min_duration*(n_stages-1-i))),                                         \
                                      int((((t_end   - (min_duration*(n_stages-1-i))) - (t_start + (min_duration*(i+1))))* density)+1)))
    return t_switches


def do_tswitches_combis(t_switches, min_duration):
    '''
    Generates all possible combinations of tswitches in a list

    Parameters:
    t_switches (list of arrays): Output from do_tswitches()
    min_duration (int or float): minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
    
    Returns:
    list
    '''
    switches_list = list(itertools.product(*t_switches))
    for i in reversed(range(len(switches_list))):                       # Go over time combination list in reversed order 
        for j in range(len(switches_list[i])-1):                        # Go within each combination and delete time combinations that make no sense (e.g. (5, 1, 3)) 
            if(switches_list[i][j]+min_duration > switches_list[i][j+1]): # the min_duration is as well incorporated here to account for minimal duration of each stage
                switches_list.pop(i)
    return(switches_list)


def do_create_times_list(n_stages, t_start, t_end, min_duration, density):
    '''
    Generates a list of all possible combinations of tswitches and includes start and end time in each array

    Parameters:
    n_stages (int): max number of stages the user wants to analyze (e.g. `2` for 2-stage)
    t_start (int or float): start time (e.g. `0`)
    t_end (int or float): end time (e.g. `24`)
    min_duration (int or float): minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
    density (int): density of time switches (e.g. `1` -> time switches are only tested at full hours; `2` -> time switches can occur every half hour); default=1
    
    Returns:
    list 
    '''
    switches = do_tswitches(n_stages, t_start, t_end, min_duration, density)

    switch_combis= do_tswitches_combis(switches, min_duration)
    for i in range(len(switch_combis)):
        switch_combis[i] = (float(t_start),) + switch_combis[i] + (float(t_end),) 
    return switch_combis


# Needed in do_brute_force_num() to skip unnecessary combinations (see below)
# from https://stackoverflow.com/questions/17837316/how-do-i-skip-a-few-iterations-in-a-for-loop
def consume(iterator, n):
    # Advance the iterator n-steps ahead. If n is none, consume entirely.

    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def do_count_combs(combi, n_stages, t_start, t_end, min_duration, density=1):
    '''
    Generates the total number of combinations

    Parameters:
    combi (list of tuples): combinations as list of tuples (e.g. `[(0, 0), (0, 1), ... , (5, 5)]`) (can be generated by itertools.product)
    n_stages (int): max number of stages the user wants to analyze (e.g. `2` for 2-stage)
    t_start (int or float): start time (e.g. `0`)
    t_end (int or float): end time (e.g. `24`)
    min_duration (int or float): minimum duration for a stage (e.g. `1`, meaning that all stages are at least 1 hour active)
    density (int): density of time switches (e.g. `1` -> time switches are only tested at full hours; `2` -> time switches can occur every half hour); default=1
    
    Returns:
    int
    '''
    t_switches = do_create_times_list(n_stages, t_start, t_end, min_duration, density)
    count=0
    for i in range(len(combi)):
        iteration = iter(range(len(t_switches)))
        for j in iteration:
            count=count+1
    return(count)


##################################################################################################################
###
###                 Analytic solution-based - Brute Force approach:
###
##################################################################################################################

def do_brute_force_ana(combis, models, n_stages, t_start, t_end, min_duration, s, density=1, indexes=[0,1,2], decimal=3):
    '''
    Function that takes as many different combinations of different parameters and times as the user provides and calculates those analytically

    Parameters:
    combi (list of tuples)      : combinations as list of tuples (e.g. [(0, 0), (0, 1), ... , (5, 5)]) (can be generated by itertools.product())
    models (list)               : all analytic models in list format
    n_stages (int)              : max number of stages the user wants to analyze (e.g. 2 for 2-stage)
    t_start (int or float)      : start time (e.g. 0)
    t_end (int or float)        : end time (e.g. 24)
    min_duration (int or float): minimum duration for a stage (e.g. 1, meaning that all stages are at least 1 hour active)
    s (array)                   : initial values for species as numpy array (e.g. array([  0.1, 100. ,   0. ]))
    density (int)               : density of time switches (e.g. 1 -> time switches are only tested at full hours; 2 -> time switches can occur every half hour); default=1
    indexes (array)             : array of indexes denoting at which index biomass, substrate, product (exactly in this order) occur
    decimal (int)               : how many decimals after rounding should be contained; Default: 3

    Returns:
    DataFrame including:
    - *combi*   = all important time points: (t_start, switching time 1, switching time 2, ..., t_end)
    - *Mod*     = modules/stages used (e.g. (0,0), this is basically an one-stage taking for both stages the same module that is the row with index 0 in the parameter 2D numpy array stored in the variable par)
    - *End_T*   = end time (when fermentation finished meaning when Substrate=0, if fermentation did not finished then t_end is set)
    - *End_X*   = biomass (at End_T)
    - *End_S*   = substrate (at End_T)
    - *End_P*   = product (at End_T)
    - *finished*= variable displaying stage at which fermentation finished as integer (e.g. 2 = finished in stage 2 (stop is set at when substrate = 0))
    - *Vol_P*   = volumetric yield (= End_P / End_T)
    - *Y_SubInput*= yield based on input substrate (= End_P / initial substrate)
    - *Y_SubUsed* = yield based on consumed substrate (= End_P / (initial substrate - End_S))
    '''
    t_switches = do_create_times_list(n_stages, t_start, t_end, min_duration, density)

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
            res=[]
            count=count+1

            finished=0
            combi=list(curr_combi)
            times=list(t_switches[j]) 
            s0=s

            while len(times)>1:
                finished= finished+1

                # Get substrate, product, biomass and end time values for the given stage 
                res = models[int(combi[0])](s0, times[0], times[1])

                ## Drop the first element from combination and time
                # e.g.          (0,1,2) and (0.0, 2.0, 3.0, 24.0)
                # will become   (  1,2) and (     2.0, 3.0, 24.0)
                combi.pop(0)
                times.pop(0)

                if((times[0]-res[-1]) < 1e-9):       # next stage (check if substrate went out before stage duration finished)
                    if(len(times)==1):              # if t_end is reached -> stop
                        res=res+[np.Inf]
                        break
                    s0= res[:-1]
                else:                               # fermentation finished before last stage        
                    res=res+[finished]
                    break

            ## DataFrame
            combi_str       = str(t_switches[j])
            combi_mod_str   = np.array(curr_combi)
            combi_mod_str   += 1
            combi_mod_str   = str(combi_mod_str.tolist())
            # Volumetric productivity
            Vol_P=res[indexes[2]]/(res[-2]-t_start)

            df_add = pl.DataFrame(data= {  # 'Name'      : [str(names[curr_combi[0]])+' - '+str(combi_t1)+' - '+str(names[curr_combi[1]])+' - '+str(combi_t2)+' -> '+str(names[curr_combi[2]])],
                                            'Index'     : [count],
                                            'Times'     : [combi_str], 
                                            'Models'    : [combi_mod_str],
                                            'End_T'     : [np.round(res[-2], decimals=decimal)],  # End time of fermentation
                                            'End_X'     : [np.round(res[indexes[0]], decimals=decimal)],  #                           corresponding Biomass value
                                            'End_S'     : [np.round(res[indexes[1]], decimals=decimal)],  #                           corresponding Substrate value
                                            'End_P'     : [np.round(res[indexes[2]], decimals=decimal)],  #                           corresponding Product value
                                            'finished'  : [str(res[-1])], # Was all substrate taken up?
                                            'Vol_P'     : [np.round(Vol_P, decimals=decimal)], # Volumetric yield
                                            'Y_SubInput': [np.round((res[indexes[2]]/s[indexes[1]]), decimals=decimal)],
                                            'Y_SubUsed' : [np.round((res[indexes[2]]/(s[indexes[1]]-res[indexes[1]])), decimals=decimal)]
                                            })
            # Append current data to df 
            df = df.extend(df_add)
            if(res[-1] == 1):        # Get new combinations, because fermentation finished in 1. Stage already
                    break
            
            if(res[-1] < n_stages):  # Skip unnecessary combinations if fermentation finished before last stage
                jj=j
                try:
                    while(t_switches[j][res[-1]-1] == t_switches[jj][res[-1]-1]):
                        if(t_switches[j][res[-1]-1] != t_switches[jj+1][res[-1]-1]):
                            break
                        consume(iteration, 1)
                        jj += 1 
                except:
                    break
    return(df.to_pandas())


##################################################################################################################
###
###                 Simulation-based - Brute Force approach:
###
##################################################################################################################

def do_brute_force_num(combis, models, n_stages, t_start, t_end, min_duration, s, events, step=0.01, density=1, arguments=None, indexes=[0,1,2], event_terminal=True, decimal=3):
    '''
    Function that takes as many different combinations of different parameters and times as the user provides and calculates those simulation/ODE-based
    
    Parameters:
    combi (list of tuples)  : combinations as list of tuples (e.g. [(0, 0), (0, 1), ... , (5, 5)]) (can be generated by itertools.product())
    models (list)           : all analytic models in list format
    n_stages (int)          : max number of stages the user wants to analyze (e.g. 2 for 2-stage)
    t_start (int or float)  : start time (e.g. 0)
    t_end (int or float)    : end time (e.g. 24)
    min_duration (int or float): minimum duration for a stage (e.g. 1, meaning that all stages are at least 1 hour active)
    s (array)               : initial values for species as numpy array (e.g. array([  0.1, 100. ,   0. ]))
    events (list of functions): events such as event_sub0 (trigger at specific time points in integration)
    step (float)            : maximal step size in integrator scipy.integrate.solve_ivp() as float and default=0.01
    density (int)           : density of time switches (e.g. 1 -> time switches are only tested at full hours; 2 -> time switches can occur every half hour); default=1
    arguments (array)       : additional auxiliary variables for models (e.g. rates)
    indexes (array)         : array of indexes denoting at which index biomass, substrate, product (exactly in this order) occur
    event_terminal (boolean) : check if integration should stop when event was triggered or not
    decimal (int)           : how many decimals after rounding should be contained; Default: 3

    Returns:
    DataFrame including:
    - *combi*   = all important time points: (t_start, switching time 1, switching time 2, ..., t_end)
    - *Mod*     = modules/stages used (e.g.(0,0), this is basically an one-stage taking for both stages the same module that is the row with index 0 in the parameter 2D numpy array stored in the variable par)
    - *End_T*   = end time (when fermentation finished meaning when Substrate=0, if fermentation did not finished then t_end is set)
    - *End_X*   = biomass (at End_T)
    - *End_S*   = substrate (at End_T)
    - *End_P*   = product (at End_T)
    - *finished*= variable displaying stage at which fermentation finished as integer (e.g. 2 = finished in stage 2 (stop is set at when substrate = 0))
    - *Vol_P*   = volumetric yield (= End_P / End_T)
    - *Y_SubInput*= yield based on input substrate (= End_P / initial substrate)
    - *Y_SubUsed* = yield based on consumed substrate (= End_P / (initial substrate - End_S))
    '''
    ## Index meanings:
    # indexes[0] = Biomass
    # indexes[1] = Substrate
    # indexes[2] = Product
    
    ## Decide for event function if the integration should stop when triggered
    if event_terminal is False:
        events[0].terminal = False
    elif event_terminal is True:
        events[0].terminal = True
    else:
        print("Set event_terminal as True or False or as list with the same length as events to specify at which events should trigger to stop integration")
        return 
    
    arguments_start=arguments
    t_switches = do_create_times_list(n_stages, t_start, t_end, min_duration, density)
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
            res=[]
            count=count+1
            finished=0
            combi=list(curr_combi)
            times=list(t_switches[j]) 
            s0=s
            arguments=arguments_start
            while len(times)>1:
                finished= finished+1
                r = sp.integrate.solve_ivp(models[int(combi[0])], t_span=[times[0], times[1]], y0=s0, t_eval=[times[1]], args=arguments, events=events, dense_output=True, max_step=step)
                
                try:    # Case in which event triggers and fermentation stopps before tend
                    res = [ r.y_events[0][0][species] for species in range(len(s)) ]
                    res = res + [r.t_events[0][0]]
                except: # Case in which fermentation runs until tend
                    try:
                        res = [ r.y[species][0] for species in range(len(s)) ]
                        res = res + [r.t[0]]
                    except: # case where a combination such as (0, 0, 20) in times was inserted and integration fails with tstart=0 and tend=0
                        res = np.zeros((len(s)+1))

                ## Drop the first element from combination and time
                # e.g.          (0,1,2) and (0.0, 2.0, 3.0, 24.0)
                # will become   (  1,2) and (     2.0, 3.0, 24.0)
                combi.pop(0)
                times.pop(0)

                if((times[0]-res[-1]) < 1e-9):       # next stage 
                    if(len(times)==1):              # if t_end is reached -> stop
                        res=res+[np.Inf]
                        break
                    s0=res[:-1]
                else:                               # fermentation finished before last stage        
                    res=res+[finished]
                    break

            ## DataFrame
            combi_str       = str(t_switches[j])
            combi_mod_str   = np.array(curr_combi)
            combi_mod_str   += 1
            combi_mod_str   = str(combi_mod_str.tolist())
            # Volumetric yield
            try:
                Vol_P=res.y_events[0][0][indexes[2]]/(res.t_events[0][0]-t_start)
            except: # if fermentation was not finished take product at t_end and devide by t_end to get vol. yield
                Vol_P=res[indexes[2]]/(res[-2]-t_start)
            
            df_add = pl.DataFrame(data= {  # 'Name'      : [str(names[curr_combi[0]])+' - '+str(combi_t1)+' - '+str(names[curr_combi[1]])+' - '+str(combi_t2)+' -> '+str(names[curr_combi[2]])],
                                            'Index'     : [count],
                                            'Times'     : [combi_str], 
                                            'Models'    : [combi_mod_str],
                                            'End_T'     : [np.round(res[-2], decimals=decimal)],  # End time of fermentation
                                            'End_X'     : [np.round(res[indexes[0]], decimals=decimal)],  #                           corresponding Biomass value
                                            'End_S'     : [np.round(res[indexes[1]], decimals=decimal)],  #                           corresponding Substrate value
                                            'End_P'     : [np.round(res[indexes[2]], decimals=decimal)],  #                           corresponding Product value
                                            'finished'  : [str(res[-1])], # Was all substrate taken up?
                                            'Vol_P'     : [np.round(Vol_P, decimals=decimal)], # Volumetric yield
                                            'Y_SubInput': [np.round((res[indexes[2]]/s[indexes[1]]), decimals=decimal)],
                                            'Y_SubUsed' : [np.round((res[indexes[2]]/(s[indexes[1]]-res[indexes[1]])), decimals=decimal)]
                                            })
            # Append current data to df 
            df = df.extend(df_add)
            if(res[-1] == 1):        # Get new combinations, because fermentation finished in 1. Stage already
                    break
            
            if(res[-1] < n_stages):  # Skip unnecessary combinations if fermentation finished before last stage
                jj=j
                try:
                    while(t_switches[j][res[-1]-1] == t_switches[jj][res[-1]-1]):
                        if(t_switches[j][res[-1]-1] != t_switches[jj+1][res[-1]-1]):
                             break
                        consume(iteration, 1)
                        jj += 1 
                except:
                     break
    return(df.to_pandas())



##################################################################################################################
###
###                 Optimizer approach:
###
##################################################################################################################


def check_constraints(row):
    '''
    Function that checks if all entries of a list are negative (then all constraints are satisfied; returns True).
    Even if only one entry is positive it will return False.
    
    Parameters:
    row (list): list of values for constraints (negative if they met constraint and positive if they did not)

    Returns:
    Boolean
    '''
    # Go over constraints and check if they were met
    for i in range(len(row)):
        if row[i]>0.0:
            return False
    return True

def do_opt_to_df(df, n_best):
    '''
    Function that takes the extracted log of the problem and outputs an interpretable dataframe with the n_best performing scores of the optimization
    
    Parameters:
    df (dataframe)  : Dataframe containing log of optimization problem
    n_best (int)    : number of best performing scores that will be returned
    
    Returns:
    list : n_best tswitches as numeric values
    list : n_best models as numeric values
    Dataframe:  containing the n_best candidates with columns:
        - Score   = is the value of the measure that was set at 'obj' to be optimized; can be 'Vol_P' for volumetric productivity or Titer for titer
        - Mod     = modules/stages used (e.g. (0,0), this is basically an one-stage taking for both stages the same module that is the row with index 0 in the parameter 2D numpy array stored in the variable 'par')
        - combi   = all important time points: (t_start, switching time 1, switching time 2, ..., t_end)
    '''
    # Adjust column names of log dataframe
    df=df.iloc[:, :4]
    df.columns = ["Score", "Models", "Times", "Constraints"]
    df=df.reset_index(drop=True)

    # Distinguish cases of optimization with and without constraints
    if len(df["Constraints"][1])>0 : # with constraints
        print("Constraints found. Dataframe is filtered first")
        filtered_rows = [row for idx, row in df.iterrows() if check_constraints(row["Constraints"])]
    
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.columns = ["Score", "Models", "Times", "Constraints"]
        filtered_df=filtered_df.sort_values(["Score"], ascending=[True]).dropna(axis=0)[:n_best]
        df["Score"] *= -1  # For optimization the objective value had to be made negative, this is now reversed back to normal
        # Incorporate tstart and tend in the times list
        combi_list = [(list(literal_eval(x))) for x in filtered_df['Times']]
        # Nested list comprehension for converting string back to list of integers and correct indices to actual module numbers (index+1)
        mod_list = [[ int(y+1) for y in list(literal_eval(x))] for x in filtered_df['Models']]
        return combi_list, mod_list, filtered_df
        
    else:     
        print("No constraints found.")                                                              # without constraints
        df=df.sort_values(["Score"], ascending=[True]).dropna(axis=0)[:n_best]
        df["Score"] *= -1  # For optimization the objective value had to be made negative, this is now reversed back to normal
        # Incorporate tstart and tend in the times list
        combi_list = [(list(literal_eval(x))) for x in df['Times']]
        # Nested list comprehension for converting string back to list of integers and correct indices to actual module numbers (index+1)
        mod_list = [[ int(y+1) for y in list(literal_eval(x))] for x in df['Models']]
        return combi_list, mod_list, df


def do_convert(opt_res, models_num, t_start, t_end, s, events, arguments=None, indexes=[0,1,2], step=0.01, event_terminal=True, decimal=3):
    '''
    Function that takes the extracted log of the problem and outputs an interpretable dataframe with the n_best performing scores of the optimization
    
    Parameters:
    opt_res (list of three entries) : Output from do_opt_to_df()
    models_num (list)               : numeric models as list
    t_start (int or float)          : start time of fermentation (e.g. 0) 
    t_end (int or float)            : end time of fermentation (e.g. 24) 
    s (array)                       : initial values for species as numpy array (e.g. array([  0.1, 100. ,   0. ])) (same values you set in optimization)
    events (list)                   : list with event functions
    arguments (list)                : list with additional parameters for models (e.g auxiliary variables)     
    indexes (array)                 : array of indexes denoting at which index biomass, substrate, product (exactly in this order) occur                  :              
    step (float)                    : maximal step size in integrator: scipy.integrate.solve_ivp() as float and default=0.01
    event_terminal (boolean)        : check if integration should stop when event was triggered or not
    decimal (int)                   : how many decimals after rounding should be contained; Default: 3

    Returns:
    dataframe
    '''
    ## Index meanings:
    # indexes[0] = Biomass
    # indexes[1] = Substrate
    # indexes[2] = Product

    ## Decide for event function if the integration should stop when triggered
    if event_terminal is False:
        events[0].terminal = False
    elif event_terminal is True:
        events[0].terminal = True
    else:
        print("Set event_terminal as True or False or as list with the same length as events to specify at which events should trigger to stop integration")
        return 

    arguments_start=arguments
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
    times_list = [([t_start]+list(x)+[t_end]) for x in opt_res[0]]
    combi_list = [[ int(y-1) for y in list(x)] for x in opt_res[1]]
    
    for i in range(len(opt_res[0])):
        res=[]
        count=count+1
        finished=0
        times=list(times_list[i]) 
        combi=list(combi_list[i])
        s0=s
        arguments=arguments_start
        while len(times)>1:
            finished= finished+1
            r = sp.integrate.solve_ivp(models_num[int(combi[0])], t_span=[times[0], times[1]], y0=s0, t_eval=[times[1]], args=arguments, events=events, dense_output=True, max_step=step)

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

            if((times[0]-res[-1]) < 1e-9):       # next stage 
                if(len(times)==1):              # if t_end is reached -> stop
                    res=res+[np.Inf]
                    break
                s0=res[0:-1]
            else:                               # fermentation finished before last stage        
                res=res+[finished]
                break

        # Volumetric yield
        try:
            Vol_P=res.y_events[0][0][indexes[2]]/(res.t_events[0][0]-t_start)
        except: # if fermentation was not finished take product at t_end and devide by t_end to get vol. yield
            Vol_P=res[indexes[2]]/(res[-2]-t_start)


        ## DataFrame
        df_add = pl.DataFrame(data= {  # 'Name'      : [str(names[curr_combi[0]])+' - '+str(combi_t1)+' - '+str(names[curr_combi[1]])+' - '+str(combi_t2)+' -> '+str(names[curr_combi[2]])],
                                        'Index'     : [count],
                                        'Times'     : [str(list(np.round(times_list[i], decimals=decimal)))], 
                                        'Models'    : [str((np.array(combi_list[i])+1).tolist())],
                                        'End_T'     : [np.round(float(res[-2]), decimals=decimal)],  # End time of fermentation
                                        'End_X'     : [np.round(res[indexes[0]], decimals=decimal)],  #                           corresponding Biomass value
                                        'End_S'     : [np.round(res[indexes[1]], decimals=decimal)],  #                           corresponding Substrate value
                                        'End_P'     : [np.round(res[indexes[2]], decimals=decimal)],  #                           corresponding Product value
                                        'finished'  : [str(res[-1])], # Was all substrate taken up?
                                        'Vol_P'     : [np.round(Vol_P, decimals=decimal)], # Volumetric productivity
                                        'Y_SubInput': [np.round((res[indexes[2]]/s[indexes[1]]), decimals=decimal)],
                                        'Y_SubUsed' : [np.round((res[indexes[2]]/(s[indexes[1]]-res[indexes[1]])), decimals=decimal)]
                                        })
        # Append current data to df 
        df = df.extend(df_add)

    return(df.to_pandas())



def f_log_decor(orig_fitness_function):
    '''
    This functions enables to store relevant values of each iteration that can then be transformed into a DataFrame later.
    for more information see: https://esa.github.io/pygmo2/tutorials/udp_meta_decorator.html

    Parameters:
    orig_fitness_function (function): objective (self, x)

    Returns:
    sol (array): the solution of the objective function
    OR
    orig_fitness_function(self, dv)
    '''
    def new_fitness_function(self, dv):
         # dv = t_switches and modules
        if hasattr(self, "dv_log"):
            sol = orig_fitness_function(self, dv)
            # Score = sol[0]
            combi=str(tuple(dv[(self.inner_problem.get_nix()-1):len(dv)]))
            times=str(tuple(dv[0:(self.inner_problem.get_nix()-1)]))
            ## This is the logger content for each iteration:
            # sol[0] = score
            # combi = combination of modules
            # times = switching times
            # sol[1] = constraints

            ## !!! weird things happening here when no constraints are provided -> the optimization algorithm takes from internal calculations some values for sol[1:] and therefore this needs to be checked and adjusted
            # print(sol[1:])
            # Solution for the problem is to distinguish cases in which extra constraints are provided and handle them correctly
            # get number of extracon
            num_extracon= self.inner_problem.get_nic() - self.inner_problem.get_nix() + 2
            if num_extracon==0:
                self.dv_log.append([sol[0]]+[combi]+[times]+[[]])
            else:
                self.dv_log.append([sol[0]]+[combi]+[times]+[sol[1:]])
            return sol
        else:
            ## Initialization of log
            self.dv_log = [dv]
            return orig_fitness_function(self, dv)
        
    return new_fitness_function



def numeric_option(models, combi, times, s0, arguments, events, step):
    '''
    This function calculates the current module/switch time combination numerically and returns the result back to objective(self,x)
    '''
    r = sp.integrate.solve_ivp(models[int(combi[0])], t_span=[times[0], times[1]], y0=s0, t_eval=[times[1]], args=arguments, events=events, dense_output=True, max_step=step)

    try:    # Case in which event triggers and fermentation stopps before tend
        res = [ r.y_events[0][0][species] for species in range(len(s0)) ]
        res = res + [r.t_events[0][0]]
    except: # Case in which fermentation runs until tend
        res = [ r.y[species][0] for species in range(len(s0)) ]
        res = res + [r.t[0]]
    return res
    
def analytic_option(models, combi, times, s0, arguments=None, events=None, step=None):
    '''
    This function calculates the current module/switch time combination analytically and returns the result back to objective(self,x)
    '''
    return models[int(combi[0])](s0, times[0], times[1])
    

def objective(self, x):
    '''
    This function is called in the optimizer class for calculating the score based on the input variables 
    (internally called and user does not need to deal with this function).
    '''
    combi=list(x[(self.max_stage-1):len(x)])
    models=self.models
    s0=self.y
    times=[self.tstart]+list(x[0:(self.max_stage-1)])+[self.tend]
    finished=0
    res=[]
    step=self.step
    arguments=self.arguments
    events=self.events
    event_terminal=self.event_terminal
    calc_option=self.calc_option

    ## Decide for event function if the integration should stop when triggered
    if events is None:
        pass
    elif event_terminal is False:
        events[0].terminal = False
    elif event_terminal is True:
        events[0].terminal = True
    else:
        print("Set event_terminal as True or False or as list with the same length as events to specify at which events should trigger to stop integration")
        return 


    while len(times)>1:
        finished= finished+1
        
        # Pass the variable to calc_option which is either the numeric_option function or the analytic_option function (see also above at the respective function)
        res=calc_option(models=models, combi=combi, times=times, s0=s0, arguments=arguments, events=events, step=step)
        ## Drop the first element from combination and time
        # e.g.          [0, 1, 2] and [0.0, 2.0, 3.0, 24.0]
        # will become   [   1, 2] and [     2.0, 3.0, 24.0]
        combi.pop(0)
        times.pop(0)

        if((times[0]-res[-1]) < 1e-9):       # next stage 
            if(len(times)==1):              # if t_end is reached -> stop
                res=res+[np.Inf]
                break
            s0=res[:-1]
        else:                               # fermentation finished before last stage        
            res=res+[finished]
            break

    ## Objective   
    score = self.objective(res, self.indexes, self.y, self.tstart, self.tend)

    ## Extra constraints (extracon)
    cons=[]
    for i in range(len(self.extracon)):
        cons=cons+[self.extracon[i](res, self.indexes, self.y, self.tstart, self.tend)]

    return score , cons 


class Optimizer:
    '''
    For the optimizer approach the package pygmo (https://esa.github.io/pygmo2/) with the IHS algorithm was used
    
    Parameter:
    s (array)           : initial values for species as numpy array (e.g. array([  0.1, 100. ,   0. ])) 
    models (list)       : numeric models as list
    indexes (array)     : array of indexes denoting at which index biomass, substrate, product (exactly in this order) occur 
    tstart (int or float) : start time of fermentation (e.g. 0)
    tend (int or float) : end time of fermentation (e.g. 24)
    max_stage (int)     : number of stages the user wants to test 
    min_duration (int or float) : minimum duration for a stage (e.g. 1, meaning that all stages are at least 1 hour active)
    objective (function) : function that will be evaluated in the objective(self, x) function defined in the optimizer (score will be calculated based on this input function) 
    optmod (string)     : modus of optimization either "max" or "min" 
    calc_option (function) : which calculation option/ type of models you have (numeric_option or analytic_option)
    arguments (list)    : list of auxiliary variables (e.g. in the form (aux,) ); Default: None  
    events (list)       : events such as event_sub0 (trigger at specific time points in integration); Default: None
    extracon (list)     : list of constraint functions; Default: []
    extracon_vals (list): values for the corresponding constraints provided in extracon; Default: [] 
    extracon_optmod (list): denotion if extracon_vals is greater or lower than the constraint value use "greater_than" or "lower_than"; Default: []
    step (float)        : maximal step size in integrator scipy.integrate.solve_ivp() as float; Default: 0.01
    event_terminal (boolean) : check if integration should stop when event was triggered or not; Default: True

    Returns:
    object based on IHS algorithm pygmo package code
    '''
    def __init__(self, s, models, indexes, tstart, tend, max_stage, min_duration, objective, optmod, calc_option=numeric_option, arguments=None, events=None, extracon=[], extracon_vals=[], extracon_optmod=[], step=0.01, event_terminal=True):
        self.y          = s                 
        self.models     = models
        self.indexes    = indexes
        self.tstart     = tstart
        self.tend       = tend
        self.max_stage  = max_stage
        self.min_duration = min_duration
        self.objective  = objective
        self.optmod     = optmod
        self.calc_option = calc_option
        self.extracon   = extracon
        self.extracon_vals = extracon_vals
        self.extracon_optmod = extracon_optmod
        self.step       = step
        self.arguments  = arguments
        self.events     = events
        self.event_terminal = event_terminal            
        self.bounds     = None
        self.do_bounds()


    def do_bounds(self):
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

    def do_ineqs(self, x):
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
        ics = self.do_ineqs(x=x) # negative values are associated to satisfied inequalities

        # Skip combinations that are not meeting the inequality constraints
        if any(n > 0 for n in ics):
            try:                # case in which extra constraints are present
                return ([0] + ics + [0]*len(self.extracon))
            except(TypeError):  # case in which NO extra constraints are present
                return ([0] + ics)

        ## Optimize
        res, cons = objective(self,x)

        for i in range(len(self.extracon_vals)):
            if self.extracon_optmod[i]=="greater_than":
                ics = ics + [(self.extracon_vals[i]-cons[i])]
            elif self.extracon_optmod[i]=="smaller_than":
                ics = ics + [(cons[i]-self.extracon_vals[i])]
            else:
                print("Put greater_than or smaller_than")
                return 
        
        # Return score as well as all constraints
        if self.optmod=="max":
            return([-res] + ics) # In pagmo minimization is always assumed; to maximize some objective function, put minus sign in front of objective.
        elif self.optmod=="min":
            return([res] + ics) # In pagmo minimization is always assumed; to maximize some objective function, put minus sign in front of objective.
        else:
            "Provide either max or min for optmod input variable"
            return 

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
            return self.max_stage-2+len(self.extracon) # this is including the inequalities of e.g. time_switch_1 being smaller than time_switch_2 and the extraconstraints 
        except(TypeError):  # case in which NO extra constraints are present
            return self.max_stage-2





##################################################################################################################
###
###                Plotting function:
###
##################################################################################################################

# Support functions for plotting
def do_tuple_to_list(df):
    '''
    Support function for do_custom_plot
    Transforming the string DataFrame columns "Times" and "Models" to numeric values as list in a list (by list comprehension)
    
    Parameters:
    df (dataframe): pandas dataframe

    Returns:
    combi_list (list)   : list of switching times
    mod_list (list)     : list of modules  
    '''
    combi_list = [literal_eval(x) for x in df['Times']]
    mod_list = [literal_eval(x) for x in df['Models']]

    return combi_list, mod_list


def do_custom_plot(df, models_num, s, title, events, arguments=None, indexes=[0,1,2], step=0.01, with_lines=False, palette="viridis", biomass_ylab='Biomass [gDW/L]', substrate_ylab='Substrate [mmol/L]', product_ylab='Product [mmol/L]'):
    '''
    This functions generates plots of biomass, substrate and product for the provided combinations of df

    Parameters:
    df (dataframe)              : dataframe you get from:
    models_num (list of functions): numeric models
    s (array)                   : start values for biomass, substrate, product
    title (string)              : title of plot displayed at the top
    events (list of functions): events such as event_sub0 (trigger at specific time points in integration)
    arguments (array)           : additional auxiliary variables for models (e.g. rates)
    indexes (array)             : array of indexes denoting at which index biomass, substrate, product (exactly in this order) occur
    step (float)                : maximal step size in integrator scipy.integrate.solve_ivp() as float and default=0.01
    with_lines (Boolean)        : Decide if additional lines in the plot should be displayed; Default: False
    palette (string)            : color palette to use; Default: "viridis"
    biomass_ylab (string)       : Biomass y label in plot; Default: 'Biomass [gDW/L]'
    substrate_ylab (string)     : Substrate y label in plot; Default: 'Substrate [mmol/L]'
    product_ylab (string)       : Product y label in plot; Default: 'Product [mmol/L]'

    Returns:
    plot object
    '''
    arguments_start=arguments
    infos = do_tuple_to_list(df)

    ## Here we do no want to terminate the integration when substrate is 0
    events[0].terminal = False

    times  = infos[0]
    combis = infos[1]
    
    dat = pd.DataFrame()
    palette = sns.color_palette(palette,n_colors=len(combis)) # https://seaborn.pydata.org/tutorial/color_palettes.html
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(5,15))
    combi_list = []
    
    ## The loop for all the different modi
    for i in range(len(combis)):
        curr_combi= combis[i]
        curr_combi = [x-1 for x in curr_combi] # Prepare for correct indexing starting from 0
        curr_times = list(times[i])

        ## Simulate ODE
        finished=0
        s0=s
        results=pd.DataFrame()
        arguments=arguments_start

        while len(curr_times)>1:
            finished= finished+1
            
            r = sp.integrate.solve_ivp(models_num[int(curr_combi[0])], t_span=[curr_times[0], curr_times[1]], y0=s0, t_eval=[curr_times[1]], args=arguments, events=events, dense_output=True, max_step=step)
            t = np.linspace(curr_times[0], curr_times[1], (int(curr_times[1])+1)*20) # (times[1]+1)*20 = Resolution
            try:
                res = [ r.y_events[0][0][species] for species in range(len(s)) ]
                res = res + [r.t_events[0][0]]
            except:
                res = [ r.y[species][0] for species in range(len(s)) ]
                res = res + [r.t[0]]


            dat_add= pd.DataFrame(data={    'Time [h]'  : t, 
                                            'Biomass'   : r.sol(t)[indexes[0]],
                                            'Substrate' : r.sol(t)[indexes[1]],
                                            'Product'   : r.sol(t)[indexes[2]]})
            results= pd.concat([results, dat_add])
            ## Drop the first element from combination and time
            # e.g.          (0,1,2) and (0.0, 2.0, 3.0, 24.0)
            # will become   (  1,2) and (     2.0, 3.0, 24.0)
            curr_combi.pop(0)
            curr_times.pop(0)
            s0=res[:-1]

        curr_times = [round(x,2) for x in list(times[i])]
        combi_str       = " ".join(str(x) for x in (combis[i],curr_times)) # For displaying in the plot

        results['Processes']=combi_str
        
        ## DataFrame
        dat= pd.concat([dat, results]) # res[1] = DataFrame with all Time, Biomass, Substrate and Product columns and times in rows
        combi_list.append(combi_str)
        

    
    # Biomass plot
    sns.lineplot(
        data=dat,
        x="Time [h]", y="Biomass",
        hue="Processes", #col=columns,
        #kind="line",# size_order=["T1", "T2"], 
        palette=palette,
        #height=5, aspect=.75, facet_kws=dict(sharex=False), 
        linewidth=0.5,
        ax=axs[0],
        legend=True
    ).set(title=title, ylim=(0, dat["Biomass"].max()*1.05))
    plt.setp(axs[0].get_legend().get_texts(), fontsize='8')
    plt.setp(axs[0].get_legend().get_title(), fontsize='10')
    plt.setp(axs[0].set_ylabel(biomass_ylab))
    
    # Substrate plot
    sns.lineplot(
        data=dat,
        x="Time [h]", y="Substrate",
        hue="Processes", #col="align",
        #kind="line",# size_order=["T1", "T2"], 
        palette=palette,
        #height=5, aspect=.75, facet_kws=dict(sharex=False), 
        linewidth=0.5,
        ax=axs[1],
        legend=True
    ).set(ylim=(0, int(s[indexes[1]])))
    plt.setp(axs[1].get_legend().get_texts(), fontsize='8')
    plt.setp(axs[1].get_legend().get_title(), fontsize='10')
    plt.setp(axs[1].set_ylabel(substrate_ylab))
    

    # Product plot
    sns.lineplot(data=dat, 
        x="Time [h]", y="Product", 
        hue="Processes", #col="align",
        #kind="line",# size_order=["T1", "T2"], 
        palette=palette,
        #height=5, aspect=.75, facet_kws=dict(sharex=False), 
        linewidth=0.5,
        ax=axs[2],
        legend=True
    ).set(ylim=(0, dat["Product"].max()*1.05))
    plt.setp(axs[2].get_legend().get_texts(), fontsize='8')
    plt.setp(axs[2].get_legend().get_title(), fontsize='10')
    plt.setp(axs[2].set_ylabel(product_ylab))
    # axs[2].axhline((s[S]*6)*0.85) # 85% theoretical product 
    
    if with_lines:
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

    return plt