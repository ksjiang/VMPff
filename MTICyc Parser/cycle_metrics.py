# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:02:41 2023

@author: Kyle
"""

import BTSff
import os

# this directory
MYDIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(MYDIR, "../General/"))
import cycle_tools

# class representing experiments recorded by MTI Cycler
class MTICycExperiment(object):
    def __init__(self):
        return
    
    # instantiate fromFile
    def fromFile(self, fileName):
        x1, x2, Y = BTSff.fromFile(fileName)
        self.metadata = [x1, x2]
        self.measurement_sequence = Y
        return
    
    def getCycleData_hc(self, cycle, half_cycle, include_rest):
        if include_rest:
            cycleData = self.measurement_sequence.loc[((self.measurement_sequence["Ns"] == cycle) | (self.measurement_sequence["Ns"] == cycle + 1)) & (self.measurement_sequence["half cycle"] == half_cycle)]
        else:
            cycleData = self.measurement_sequence.loc[(self.measurement_sequence["Ns"] == cycle) & (self.measurement_sequence["half cycle"] == half_cycle)]
        
        return cycleData
    
    
class MTICycMODE1CyclingExperiment(MTICycExperiment, cycle_tools.MODE1CyclingExperiment):
    def __init__(self, area):
        MTICycExperiment.__init__(self)
        cycle_tools.MODE1CyclingExperiment.__init__(self, area, REST = (1, 0), CYCLE_PLATING = (2, 1), CYCLE_STRIPPING = (4, 2))
        return
    
    
class MTICycPNNLCyclingExperiment(MTICycExperiment, cycle_tools.PNNLCyclingExperiment):
    def __init__(self, area):
        MTICycExperiment.__init__(self)
        cycle_tools.PNNLCyclingExperiment.__init__(self, area, REST = (1, 0), INITIAL_PLATING = (2, 1), INITIAL_STRIPPING = (4, 2), TEST_PLATING = (6, 3), SHORT_CYCLE_PLATING = (8, 4), SHORT_CYCLE_STRIPPING = (10, 5), TEST_STRIPPING = (13, 22), NUM_SHORT_CYCLES = 10)
        return
    
    