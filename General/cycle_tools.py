# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:36:28 2023

@author: Kyle
"""

import os
MYDIR = os.path.dirname(__file__)

import pandas as pd

import sys
sys.path.append(os.path.join(MYDIR, "../General/"))
import common

sec2hr = lambda time_delta: time_delta / 3600.
specCapacity = lambda capacity, area: capacity / area

# common indices to reference first and last datapoints in some ordered series
FIRST, LAST = 0, -1

# custom exception type for unrecognized instrument
class UnrecognizedInstrumentException(Exception):
    pass

def capacityDiff(cycleData):
    return cycleData.iloc[LAST]["capacity"] - cycleData.iloc[FIRST]["capacity"]

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