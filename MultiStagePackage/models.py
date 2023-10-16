
#########################################################
###
### Models
###

import numpy as np

###  Indexing
X_0,S_0,P_0 = (i for i in range(3))               
X  ,S  ,P = (i for i in range(3))             
r_S, r_P, mu = (i for i in range(3))  

## Numeric ODE models:
# y  = start values for biomass, substrate, product [numpy array]
def WT_anaerob_growth(t, y):
    # Measurement-based parameters: 
    mu  = 0.13
    r_S = 16.13
    r_P = 26.11

    # Check for values smaller 0
    for i in range(len(y)):
        if(y[i]<0.0):
            y[i] = 0.0
    
    if(y[S]>0.001):
        dXdt =   mu*y[X]
        dSdt = - r_S*y[X]
        dPdt =   r_P*y[X]
    else:
        dXdt = 0
        dSdt = 0
        dPdt = 0
    if(dXdt<0.0):
        print(dXdt)
    
    dydt = [dXdt, dSdt, dPdt]
    return(dydt)

def LC_anaerob_growth(t, y):
    # Measurement-based parameters:
    mu  = 0.06
    r_S = 22.86
    r_P = 41.25  

    # Check for values smaller 0
    for i in range(len(y)):
        if(y[i]<0.0):
            y[i] = 0.0
    
    if(y[S]>0.001):
        dXdt =   mu*y[X]
        dSdt = - r_S*y[X]
        dPdt =   r_P*y[X]
    else:
        dXdt = 0
        dSdt = 0
        dPdt = 0
    if(dXdt<0.0):
        print(dXdt)
    
    dydt = [dXdt, dSdt, dPdt]
    return(dydt)

def WT_aerob_growth(t, y):
    # Measurement-based parameters:
    mu  = 0.52
    r_S = 7.62
    r_P = 0.12  

    # Check for values smaller 0
    for i in range(len(y)):
        if(y[i]<0.0):
            y[i] = 0.0
    
    if(y[S]>0.001):
        dXdt =   mu*y[X]
        dSdt = - r_S*y[X]
        dPdt =   r_P*y[X]
    else:
        dXdt = 0
        dSdt = 0
        dPdt = 0
    if(dXdt<0.0):
        print(dXdt)
    
    dydt = [dXdt, dSdt, dPdt]
    return(dydt)

def LC_aerob_growth(t, y):
    # Measurement-based parameters:
    mu  = 0.50
    r_S = 6.08
    r_P = 0.08

    # Check for values smaller 0
    for i in range(len(y)):
        if(y[i]<0.0):
            y[i] = 0.0
    
    if(y[S]>0.001):
        dXdt =   mu*y[X]
        dSdt = - r_S*y[X]
        dPdt =   r_P*y[X]
    else:
        dXdt = 0
        dSdt = 0
        dPdt = 0
    if(dXdt<0.0):
        print(dXdt)
    
    dydt = [dXdt, dSdt, dPdt]
    return(dydt)

def WT_anaerob_growth_arrest(t, y):
    # Measurement-based parameters:
    mu  = 0.00
    r_S = 0.86
    r_P = 1.42

    # Check for values smaller 0
    for i in range(len(y)):
        if(y[i]<0.0):
            y[i] = 0.0
    
    if(y[S]>0.001):
        dXdt =   mu*y[X]
        dSdt = - r_S*y[X]
        dPdt =   r_P*y[X]
    else:
        dXdt = 0
        dSdt = 0
        dPdt = 0
    if(dXdt<0.0):
        print(dXdt)
    
    dydt = [dXdt, dSdt, dPdt]
    return(dydt)

def LC_anaerob_growth_arrest(t, y):
    # Measurement-based parameters:
    mu  = 0.00
    r_S = 9.37
    r_P = 17.61

    # Check for values smaller 0
    for i in range(len(y)):
        if(y[i]<0.0):
            y[i] = 0.0
    
    if(y[S]>0.001):
        dXdt =   mu*y[X]
        dSdt = - r_S*y[X]
        dPdt =   r_P*y[X]
    else:
        dXdt = 0
        dSdt = 0
        dPdt = 0
    if(dXdt<0.0):
        print(dXdt)
    
    dydt = [dXdt, dSdt, dPdt]
    return(dydt)


## Analytical models:
# y  = start values for biomass, substrate, product [numpy array]
# tstart = start time [integer]
# tend   = end time   [integer] 
def WT_anaerob_growth_analyt(y, tstart, tend):
    # Measurement-based parameters:
    mu  = 0.13
    r_S = 16.13
    r_P = 26.11

    if (mu>=1e-3):                                               ### with growth (mu is positive)
        resX=y[X]*np.exp(mu*(tend-tstart))                       # Biomass
        resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative
            tend=np.log(y[S]*(mu/r_S)/y[X]+1)/mu+tstart          # calculate tend where S=0
            resX=y[X]*np.exp(mu*(tend-tstart))                   # Get Biomass at time where S=0
            resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)   # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+r_P/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Product 

    else:                                                        ### no growth (mu = 0)
        resX=y[X]                                                # Biomass (is X0 because mu=0)
        resS=y[S]-y[X]*r_S*(tend-tstart)                         # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative 
            tend=(y[S]/y[X])/r_S+tstart                          # calculate tend where S=0
            resS=y[S]-y[X]*r_S*(tend-tstart)                     # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+y[X]*r_P*(tend-tstart)                         # Product
    return([resX, resS, resP, tend])

def LC_anaerob_growth_analyt(y, tstart, tend):
    # Measurement-based parameters:
    mu  = 0.06
    r_S = 22.86
    r_P = 41.25 

    if (mu>=1e-3):                                               ### with growth (mu is positive)
        resX=y[X]*np.exp(mu*(tend-tstart))                       # Biomass
        resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative
            tend=np.log(y[S]*(mu/r_S)/y[X]+1)/mu+tstart          # calculate tend where S=0
            resX=y[X]*np.exp(mu*(tend-tstart))                   # Get Biomass at time where S=0
            resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)   # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+r_P/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Product 

    else:                                                        ### no growth (mu = 0)
        resX=y[X]                                                # Biomass (is X0 because mu=0)
        resS=y[S]-y[X]*r_S*(tend-tstart)                         # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative 
            tend=(y[S]/y[X])/r_S+tstart                          # calculate tend where S=0
            resS=y[S]-y[X]*r_S*(tend-tstart)                     # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+y[X]*r_P*(tend-tstart)                         # Product
    return([resX, resS, resP, tend])

def WT_aerob_growth_analyt(y, tstart, tend):
    # Measurement-based parameters:
    mu  = 0.52
    r_S = 7.62
    r_P = 0.12
    
    if (mu>=1e-3):                                               ### with growth (mu is positive)
        resX=y[X]*np.exp(mu*(tend-tstart))                       # Biomass
        resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative
            tend=np.log(y[S]*(mu/r_S)/y[X]+1)/mu+tstart          # calculate tend where S=0
            resX=y[X]*np.exp(mu*(tend-tstart))                   # Get Biomass at time where S=0
            resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)   # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+r_P/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Product 

    else:                                                        ### no growth (mu = 0)
        resX=y[X]                                                # Biomass (is X0 because mu=0)
        resS=y[S]-y[X]*r_S*(tend-tstart)                         # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative 
            tend=(y[S]/y[X])/r_S+tstart                          # calculate tend where S=0
            resS=y[S]-y[X]*r_S*(tend-tstart)                     # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+y[X]*r_P*(tend-tstart)                         # Product
    return([resX, resS, resP, tend])

def LC_aerob_growth_analyt(y, tstart, tend):
    # Measurement-based parameters:
    mu  = 0.50
    r_S = 6.08
    r_P = 0.08

    if (mu>=1e-3):                                               ### with growth (mu is positive)
        resX=y[X]*np.exp(mu*(tend-tstart))                       # Biomass
        resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative
            tend=np.log(y[S]*(mu/r_S)/y[X]+1)/mu+tstart          # calculate tend where S=0
            resX=y[X]*np.exp(mu*(tend-tstart))                   # Get Biomass at time where S=0
            resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)   # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+r_P/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Product 

    else:                                                        ### no growth (mu = 0)
        resX=y[X]                                                # Biomass (is X0 because mu=0)
        resS=y[S]-y[X]*r_S*(tend-tstart)                         # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative 
            tend=(y[S]/y[X])/r_S+tstart                          # calculate tend where S=0
            resS=y[S]-y[X]*r_S*(tend-tstart)                     # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+y[X]*r_P*(tend-tstart)                         # Product
    return([resX, resS, resP, tend])    

def WT_anaerob_growth_arrest_analyt(y, tstart, tend):
    # Measurement-based parameters:
    mu  = 0.00
    r_S = 0.86
    r_P = 1.42

    if (mu>=1e-3):                                               ### with growth (mu is positive)
        resX=y[X]*np.exp(mu*(tend-tstart))                       # Biomass
        resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative
            tend=np.log(y[S]*(mu/r_S)/y[X]+1)/mu+tstart          # calculate tend where S=0
            resX=y[X]*np.exp(mu*(tend-tstart))                   # Get Biomass at time where S=0
            resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)   # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+r_P/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Product 

    else:                                                        ### no growth (mu = 0)
        resX=y[X]                                                # Biomass (is X0 because mu=0)
        resS=y[S]-y[X]*r_S*(tend-tstart)                         # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative 
            tend=(y[S]/y[X])/r_S+tstart                          # calculate tend where S=0
            resS=y[S]-y[X]*r_S*(tend-tstart)                     # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+y[X]*r_P*(tend-tstart)                         # Product
    return([resX, resS, resP, tend])

def LC_anaerob_growth_arrest_analyt(y, tstart, tend):
    # Measurement-based parameters:
    mu  = 0.00
    r_S = 9.37
    r_P = 17.61

    if (mu>=1e-3):                                               ### with growth (mu is positive)
        resX=y[X]*np.exp(mu*(tend-tstart))                       # Biomass
        resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative
            tend=np.log(y[S]*(mu/r_S)/y[X]+1)/mu+tstart          # calculate tend where S=0
            resX=y[X]*np.exp(mu*(tend-tstart))                   # Get Biomass at time where S=0
            resS=y[S]-r_S/mu*y[X]*(np.exp(mu*(tend-tstart))-1)   # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+r_P/mu*y[X]*(np.exp(mu*(tend-tstart))-1)       # Product 

    else:                                                        ### no growth (mu = 0)
        resX=y[X]                                                # Biomass (is X0 because mu=0)
        resS=y[S]-y[X]*r_S*(tend-tstart)                         # Substrate
    
        if (resS<0.0):                                           ## If Substrate is negative 
            tend=(y[S]/y[X])/r_S+tstart                          # calculate tend where S=0
            resS=y[S]-y[X]*r_S*(tend-tstart)                     # Get Substrate at time where S=0 (as proof/control; should result in 0)
        
        resP=y[P]+y[X]*r_P*(tend-tstart)                         # Product
    return([resX, resS, resP, tend])

