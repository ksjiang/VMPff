# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:02:41 2023

@author: Kyle
"""

import BTSff
import pandas as pd
import numpy as np
import os

# this directory
MYDIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(MYDIR, "../General/"))
from cycle_tools import *

# Mode 1 (constant-capacity cycling)
# TODO: update this for BTS
# step numbers in the sequence - Biologic
MODE1_REST = (1, 0)
MODE1_CYCLE_PLATING = (2, 1)
MODE1_CYCLE_STRIPPING = (4, 2)

# Aurbach protocol
# step numbers in the PNNL sequence - BTS
PNNL_REST = (1, 0)
PNNL_INITIAL_PLATING = (2, 1)
PNNL_INITIAL_STRIPPING = (4, 2)
PNNL_TEST_PLATING = (6, 3)
PNNL_SHORT_CYCLE_PLATING = (8, 4)
PNNL_SHORT_CYCLE_STRIPPING = (10, 5)
PNNL_TEST_STRIPPING = (13, 22)
# constants
PNNL_NUM_SHORT_CYCLES = 10

# methods to calculate the Coulombic efficiency
#===================================================================================================
def MODE1_CE(measurement_sequence, area):
    # we can only calculate CEs up to the last full cycle
    CEs = []
    cyc_num = 0
    while True:
        plating = VvsCapacity_hc(measurement_sequence, area, MODE1_CYCLE_PLATING[0], MODE1_CYCLE_PLATING[1] + 2 * cyc_num)
        if len(plating["capacity"]) == 0:
            break
        
        stripping = VvsCapacity_hc(measurement_sequence, area, MODE1_CYCLE_STRIPPING[0], MODE1_CYCLE_STRIPPING[1] + 2 * cyc_num)
        if len(stripping["capacity"]) == 0:
            break
        
        # calculate CE
        CEs.append(capacityDiff(stripping) / capacityDiff(plating))
        cyc_num += 1
        
    return np.array(CEs)

def PNNL_aurbach_CE(measurement_sequence, area, hc_step = 0):
    # calculate initial cycle CE
    initialCE = capacityDiff(VvsCapacity_hc(measurement_sequence, area, *PNNL_INITIAL_STRIPPING, hc_step = hc_step)) / capacityDiff(VvsCapacity_hc(measurement_sequence, area, *PNNL_INITIAL_PLATING, hc_step = hc_step))
    
    # calculate test cycle CE
    plating_caps, stripping_caps = [], []
    # initial plating
    plating_caps.append(capacityDiff(VvsCapacity_hc(measurement_sequence, area, *PNNL_TEST_PLATING, hc_step = hc_step)))
    #cycling
    for i in range(PNNL_NUM_SHORT_CYCLES):
        plating_caps.append(capacityDiff(VvsCapacity_hc(measurement_sequence, area, PNNL_SHORT_CYCLE_PLATING[0], PNNL_SHORT_CYCLE_PLATING[1] + 2 * i, hc_step = hc_step)))
        stripping_caps.append(capacityDiff(VvsCapacity_hc(measurement_sequence, area, PNNL_SHORT_CYCLE_STRIPPING[0], PNNL_SHORT_CYCLE_STRIPPING[1] + 2 * i, hc_step = hc_step)))
        
    # final stripping
    stripping_caps.append(capacityDiff(VvsCapacity_hc(measurement_sequence, area, *PNNL_TEST_STRIPPING, hc_step = hc_step)))
    testCE = sum(stripping_caps) / sum(plating_caps)
    return initialCE, testCE

#===================================================================================================

def PNNL_aurbach_timeseries(measurement_sequence):
    # we are interested in all cycles except for the first (rest step)
    cycleData = measurement_sequence.loc[measurement_sequence["Ns"] > PNNL_REST[0]]
    return cycleData[["time", "Ewe"]].reset_index()

def getCycleData_hc(measurement_sequence, cycle, half_cycle, include_rest):
    if include_rest:
        cycleData = measurement_sequence.loc[((measurement_sequence["Ns"] == cycle) | (measurement_sequence["Ns"] == cycle + 1)) & (measurement_sequence["half cycle"] == half_cycle)]
    else:
        cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle)]
        
    return cycleData

def VvsT_hc(measurement_sequence, cycle, half_cycle, relative = False, include_rest = False, hc_step = 0):
    cycleData = getCycleData_hc(measurement_sequence, cycle, half_cycle + hc_step, include_rest = include_rest)
    t, V = np.array(cycleData["time"]), np.array(cycleData["Ewe"])
    t = sec2hr(t)
    if relative and len(t) > 0:
        # calculate time relative to cycle start
        t = t - t[FIRST]
        
    return pd.DataFrame({
            "time": t, 
            "voltage": V, 
            })

def VvsCapacity_hc(measurement_sequence, area, cycle, half_cycle, relative = True, include_rest = False, hc_step = 0):
    cycleData = getCycleData_hc(measurement_sequence, cycle, half_cycle + hc_step, include_rest = include_rest)        
    Q, V = np.array(cycleData["Q-Q0"]), np.array(cycleData["Ewe"])
    # compute specific capacity
    Q = specCapacity(Q, area)
    if relative and len(Q) > 0:
        Q = np.abs(Q - Q[FIRST])
        
    return pd.DataFrame({
            "capacity": Q, 
            "voltage": V, 
            })
