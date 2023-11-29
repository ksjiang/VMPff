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
# step numbers in the PNNL sequence - BTS
PNNL_REST_BTS = (1, 0)
PNNL_INITIAL_PLATING_BTS = (2, 1)
PNNL_INITIAL_STRIPPING_BTS = (4, 2)
PNNL_TEST_PLATING_BTS = (6, 3)
PNNL_SHORT_CYCLE_PLATING_BTS = (8, 4)
PNNL_SHORT_CYCLE_STRIPPING_BTS = (10, 5)
PNNL_TEST_STRIPPING_BTS = (13, 22)
# constants
PNNL_NUM_SHORT_CYCLES = 10

sec2hr = lambda time_delta: time_delta / 3600.

# custom exception type for unrecognized instrument
class UnrecognizedInstrumentException(Exception):
    pass

# returns the leading edge (smaller index) of changes
# expects numpy array, so convert using to_numpy() if using pandas
def findStepChanges(series):
    return np.where(np.diff(series) != 0)[0]

def findHalfStepChanges(series, triggers):
    changes = []
    for trigger in triggers:
        hot = np.where(series == trigger)[0]
        leading_edge_idcs = np.where(np.diff(hot) != 1)[0] + 1
        # subtract one to get leading edge index (don't subtract if already 0)
        hot_leading_edges = np.maximum(0, hot[leading_edge_idcs] - 1)
        # treat the first instance
        # subtract one to get leading edge index (don't subtract if already 0)
        first = max([0, hot[0] - 1])
        if first not in hot_leading_edges:
            hot_leading_edges = np.concatenate([[first], hot_leading_edges], axis = 0)
            
        changes.append(hot_leading_edges)
        
    return np.sort(np.concatenate(changes, axis = 0))

# accumulates a series by referencing step changes
# expects numpy array
def accumulateSeriesSteps(series, change_indices):
    # new series
    r = np.zeros(len(series))
    # accumulator and mark
    a = 0.
    mark = 0
    for change_index in change_indices:
        r[mark: change_index + 1] = a + series[mark: change_index + 1]
        # update accumulator and mark
        a += series[change_index]
        mark = change_index + 1
        
    # finish the series
    r[mark: ] = a + series[mark: ]
    return r

def fillCount(series_size, change_indices):
    # new series
    r = np.zeros(series_size, dtype = int)
    a = 0
    mark = 0
    for change_index in change_indices:
        r[mark: change_index + 1] = a
        a += 1
        mark = change_index + 1
        
    # finish the series
    r[mark: ] = a
    return r

def capacityDiff(cycleData):
    return cycleData.iloc[LAST]["capacity"] - cycleData.iloc[FIRST]["capacity"]


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

def PNNL_aurbach_timeseries(measurement_sequence, instrument = "Biologic"):
    # we are interested in all cycles except for the first (rest step)
    if instrument == "Biologic":
        cycleData = measurement_sequence.loc[measurement_sequence["Ns"] > PNNL_REST[0]]
    elif instrument == "BTS":
        cycleData = measurement_sequence.loc[measurement_sequence["Ns"] > PNNL_REST_BTS[0]]
    else: raise UnrecognizedInstrumentException
    
    return cycleData[["time", "Ewe"]].reset_index()

def getCycleData_hc(measurement_sequence, cycle, half_cycle, include_rest, instrument):
    if include_rest:
        if instrument == "Biologic":
            cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle)]
        elif instrument == "BTS":
            cycleData = measurement_sequence.loc[((measurement_sequence["Ns"] == cycle) | (measurement_sequence["Ns"] == cycle + 1)) & (measurement_sequence["half cycle"] == half_cycle)]
        else: raise UnrecognizedInstrumentException
        
    else:
        if instrument == "Biologic":
            cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle) & (measurement_sequence["mode"] != VMPff.REST_MODE)]
        elif instrument == "BTS":
            cycleData = measurement_sequence.loc[(measurement_sequence["Ns"] == cycle) & (measurement_sequence["half cycle"] == half_cycle)]
        else: raise UnrecognizedInstrumentException
        
    return cycleData

def VvsT_hc(measurement_sequence, cycle, half_cycle, relative = False, include_rest = False, hc_step = 0, instrument = "Biologic"):
    cycleData = getCycleData_hc(measurement_sequence, cycle, half_cycle + hc_step, include_rest = include_rest, instrument = instrument)
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

def VvsCapacity_hc(measurement_sequence, area, cycle, half_cycle, relative = True, include_rest = False, hc_step = 0, instrument = "Biologic"):
    cycleData = getCycleData_hc(measurement_sequence, cycle, half_cycle + hc_step, include_rest = include_rest, instrument = instrument)        
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
