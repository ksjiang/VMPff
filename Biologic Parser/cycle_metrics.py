# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:02:41 2023

@author: Kyle
"""

import VMPff
import pandas as pd
import numpy as np
import os

# this directory
MYDIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(MYDIR, "../General/"))
from cycle_tools import *

# Mode 1 (constant-capacity cycling)
# step numbers in the sequence - Biologic
MODE1_REST = (0, 0)
MODE1_CYCLE_PLATING = (1, 0)
MODE1_CYCLE_STRIPPING = (2, 1)

# Aurbach protocol
# step numbers in the PNNL sequence - Biologic
PNNL_REST = (0, 0)
PNNL_INITIAL_PLATING = (1, 0)
PNNL_INITIAL_STRIPPING = (2, 1)
PNNL_TEST_PLATING = (3, 2)
PNNL_SHORT_CYCLE_PLATING = (4, 3)
PNNL_SHORT_CYCLE_STRIPPING = (5, 4)
PNNL_TEST_STRIPPING = (6, 23)
# constants
PNNL_NUM_SHORT_CYCLES = 10

# methods to calculate the Coulombic efficiency
#===================================================================================================
def MODE1_CE(measurement_sequence, area, hc_step = 0):
    # we can only calculate CEs up to the last full cycle
    # TODO: there is probably a better way to do this, by extracting the total number of half cycles
    CEs = []
    cyc_num = 0
    while True:
        plating = VvsCapacity_hc(measurement_sequence, area, MODE1_CYCLE_PLATING[0], MODE1_CYCLE_PLATING[1] + 2 * cyc_num, hc_step = hc_step)
        if not checkHalfCycle(plating): break
        stripping = VvsCapacity_hc(measurement_sequence, area, MODE1_CYCLE_STRIPPING[0], MODE1_CYCLE_STRIPPING[1] + 2 * cyc_num, hc_step = hc_step)
        if not checkHalfCycle(stripping): break
        # calculate CE for this cycle
        CEs.append(capacityDiff(stripping) / capacityDiff(plating))
        cyc_num += 1
        
    return np.array(CEs)

def PNNL_aurbach_CE(measurement_sequence, area, hc_step = 0):
    # calculate initial cycle CE
    initial_plating = VvsCapacity_hc(measurement_sequence, area, *PNNL_INITIAL_STRIPPING, hc_step = hc_step)
    checkHalfCycle(initial_plating, throw_error = True)
    initial_stripping = VvsCapacity_hc(measurement_sequence, area, *PNNL_INITIAL_PLATING, hc_step = hc_step)
    checkHalfCycle(initial_stripping, throw_error = True)
    initialCE = capacityDiff(initial_stripping) / capacityDiff(initial_plating)
    
    # calculate test cycle CE
    plating_caps, stripping_caps = [], []
    # test plating
    test_plating = VvsCapacity_hc(measurement_sequence, area, *PNNL_TEST_PLATING, hc_step = hc_step)
    checkHalfCycle(test_plating, throw_error = True)
    plating_caps.append(capacityDiff(test_plating))
    #cycling
    for i in range(PNNL_NUM_SHORT_CYCLES):
        short_plating = VvsCapacity_hc(measurement_sequence, area, PNNL_SHORT_CYCLE_PLATING[0], PNNL_SHORT_CYCLE_PLATING[1] + 2 * i, hc_step = hc_step)
        checkHalfCycle(short_plating, throw_error = True)
        plating_caps.append(capacityDiff(short_plating))
        short_stripping = VvsCapacity_hc(measurement_sequence, area, PNNL_SHORT_CYCLE_STRIPPING[0], PNNL_SHORT_CYCLE_STRIPPING[1] + 2 * i, hc_step = hc_step)
        checkHalfCycle(short_stripping, throw_error = True)
        stripping_caps.append(capacityDiff(short_stripping))
        
    # test stripping
    test_stripping = VvsCapacity_hc(measurement_sequence, area, *PNNL_TEST_STRIPPING, hc_step = hc_step)
    checkHalfCycle(test_stripping, throw_error = True)
    stripping_caps.append(capacityDiff(test_stripping))
    testCE = sum(stripping_caps) / sum(plating_caps)
    return initialCE, testCE

#===================================================================================================

# get all data after rest step
PNNL_aurbach_timeseries = lambda measurement_sequence: getVvsTAfterStep(measurement_sequence, PNNL_REST[0])

def getCycleData_hc(measurement_sequence, cycle, half_cycle, include_rest):
    if include_rest:
        cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle)]
    else:
        cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle) & (measurement_sequence["mode"] != VMPff.REST_MODE)]
        
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
    