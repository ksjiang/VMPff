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
import common

# common indices to reference first and last datapoints in some ordered series
FIRST, LAST = 0, -1

# Mode 1 (constant-capacity cycling)
# step numbers in the sequence
MODE1_REST = (0, 0)
MODE1_CYCLE_PLATING = (1, 0)
MODE1_CYCLE_STRIPPING = (2, 1)

# Aurbach protocol
# step numbers in the PNNL sequence
PNNL_REST = (0, 0)
PNNL_INITIAL_PLATING = (1, 0)
PNNL_INITIAL_STRIPPING = (2, 1)
PNNL_TEST_PLATING = (3, 2)
PNNL_SHORT_CYCLE_PLATING = (4, 3)
PNNL_SHORT_CYCLE_STRIPPING = (5, 4)
PNNL_TEST_STRIPPING = (6, 23)
# constants
PNNL_NUM_SHORT_CYCLES = 10

def capacityDiff(cycleData):
    return cycleData.iloc[LAST]["capacity"] - cycleData.iloc[FIRST]["capacity"]

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

def PNNL_aurbach_timeseries(measurement_sequence):
    # we are interested in all cycles except for the first (rest step)
    cycleData = measurement_sequence.loc[measurement_sequence["Ns"] > PNNL_REST[0]]
    return cycleData[["time", "Ewe"]]

def sec2hr(time_delta):
    return time_delta / 3600

def VvsT_hc(measurement_sequence, cycle, half_cycle, relative = False, include_rest = False, hc_step = 0):
    half_cycle += hc_step
    if include_rest:
        cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle)]
    else:
        cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle) & (measurement_sequence["mode"] != VMPff.REST_MODE)]
        
    t, V = np.array(cycleData["time"]), np.array(cycleData["Ewe"])
    t = sec2hr(t)
    if relative and len(t) > 0:
        # calculate time relative to cycle start
        t = t - t[FIRST]
        
    return pd.DataFrame({
            "time": t, 
            "voltage": V, 
            })

def specCapacity(capacity, area):
    return capacity / area

def VvsCapacity_hc(measurement_sequence, area, cycle, half_cycle, relative = True, include_rest = False, hc_step = 0):
    half_cycle += hc_step
    if include_rest:
        cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle)]
    else:
        cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle) & (measurement_sequence["mode"] != VMPff.REST_MODE)]
        
    Q, V = np.array(cycleData["Q-Q0"]), np.array(cycleData["Ewe"])
    # compute specific capacity
    Q = specCapacity(Q, area)
    if relative and len(Q) > 0:
        Q = np.abs(Q - Q[FIRST])
        
    return pd.DataFrame({
            "capacity": Q, 
            "voltage": V, 
            })

# concatenates data from multiple half-cycles
def stitchHalfCycles(half_cycles, add_breaks = False):
    assert len(half_cycles) > 0, "Empty data"
    # get columns and check that data are homogeneous
    data_columns = half_cycles[FIRST].columns
    for half_cycle in half_cycles[FIRST + 1: ]:
        assert (half_cycle.columns == data_columns).all(), "Unmatching data columns"
        
    if add_breaks:
        # create the break item
        brk = {}
        for data_column in data_columns:
            brk[data_column] = [None]
            
        brk = pd.DataFrame(brk)
        # add breaks between every pair of data series
        half_cycles = common.intersperse(half_cycles, brk)
        
    return pd.concat(half_cycles, ignore_index = True)