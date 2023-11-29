# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 01:49:29 2023

@author: Kyle
"""

import os

import numpy as np
import pandas as pd

MYDIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(MYDIR, "../General/"))
import common

FILE_HEADER = b"NEWARE"
YEAR_SLEN, MONTH_SLEN, DAY_SLEN = 4, 2, 2
BTS_VERSION_STRING_OFFS = 0x70
BTS_VERSION_STRING_SLEN = 30
CHANNEL_INFO_OFFS = 0x82b
USERNAME_OFFS = 0x876
USERNAME_LEN = 15
BATCH_LEN = 20
MEMO_LEN = 100

UA2MA = lambda x: x * 1E-3
TMV2V = lambda x: x * 1E-4
UW2W = lambda x: x * 1E-6
S2HR = lambda x: x / 3600.
UAS2MAH = lambda x: S2HR(UA2MA(x))
UWS2WH = lambda x: S2HR(UW2W(x))

BTS_VARIABLE_LIST = [
        "status", 
        "record_no", 
        "cycle number", 
        "Ns", 
        "mode", 
        "step_time", 
        "voltage", 
        "current", 
        "temperature", 
        "Q-Q0", 
        "Energy", 
        "clock_time", 
        "checksum", 
        ]

BTS_VARIABLE_FILE_DTYPES = [
        np.uint8, 
        np.uint32, 
        np.uint32, 
        np.uint8, 
        np.uint8, 
        np.uint32, 
        np.int32, 
        np.int32, 
        np.int64, 
        np.int64, 
        np.int64, 
        np.uint64, 
        np.uint32, 
        ]

BTS_RECORD_INFO_SPEC = list(zip(BTS_VARIABLE_LIST, BTS_VARIABLE_FILE_DTYPES))
BTS_RECORD_INFO = np.dtype(BTS_RECORD_INFO_SPEC)

BTS_STEP_TYPES = {
        1: "CC_charge", 
        2: "CC_discharge", 
        4: "Rest", 
        5: "Loop", 
        6: "Stop", 
        }

def fromFile(fileName):
    ptr = common.pointer()
    with open(fileName, "rb") as f:
        contents = f.read()
        
    # check magic bytes
    common.checkField(contents, ptr, FILE_HEADER)
    # initialize header info structure
    header_info = dict()
    # get date YYYYMMDD
    date = dict()
    date["year"] = int(common.getRaw(contents, ptr, YEAR_SLEN))
    date["month"] = int(common.getRaw(contents, ptr, MONTH_SLEN))
    date["day"] = int(common.getRaw(contents, ptr, DAY_SLEN))
    header_info["date"] = date
    
    ptr.setValue(BTS_VERSION_STRING_OFFS)
    header_info["version"] = str(common.getRaw(contents, ptr, BTS_VERSION_STRING_SLEN), "utf-8")
    
    ptr.setValue(CHANNEL_INFO_OFFS)
    header_info["machine"] = common.getField(contents, ptr, *common.UINT8)
    header_info["version"] = common.getField(contents, ptr, *common.UINT8)
    
    ptr.setValue(USERNAME_OFFS)
    header_info["username"] = str(common.getRaw(contents, ptr, USERNAME_LEN), "utf-8")
    header_info["batch"] = str(common.getRaw(contents, ptr, BATCH_LEN), "utf-8")
    header_info["memo"]= str(common.getRaw(contents, ptr, MEMO_LEN), "utf-8")
    
    # this brings us to the step definitions
    # extract the step definitions from header
    # TODO: there should be a better way to figure out where this header ends
    step_infos = [None]
    step_id = 1
    while True:
        assert step_id < 0xff, "Too many steps encountered, possible parsing error"
        # initialize step info dict
        step_info = dict()
        if common.getFieldFlat(contents, ptr.getValue(), *common.UINT8) != step_id:
            break
        
        # update pointer
        ptr.add(common.UINT8[1])
        step_info["Ns"] = step_id
        step_type = common.getField(contents, ptr, *common.UINT8)
        assert step_type in BTS_STEP_TYPES, "Unrecognized step type"
        step_info["mode"] = BTS_STEP_TYPES[step_type]
        if step_info["mode"] in ["CC_charge", "CC_discharge"]:
            step_info["current"] = UA2MA(common.getField(contents, ptr, *common.INT32))
            step_info["time"] = common.getField(contents, ptr, *common.INT32)
            step_info["voltage"] = TMV2V(common.getField(contents, ptr, *common.INT32))
            ptr.add(2 * common.INT32[1])
        elif step_info["mode"] == "Rest":
            step_info["time"] = common.getField(contents, ptr, *common.INT32)
            ptr.add(4 * common.INT32[1])
        elif step_info["mode"] == "Loop":
            step_info["target"] = common.getField(contents, ptr, *common.INT32)
            step_info["repeats"] = common.getField(contents, ptr, *common.INT32)
            ptr.add(3 * common.INT32[1])
        elif step_info["mode"] == "Stop":
            # no parameters
            ptr.add(5 * common.INT32[1])
            
        step_infos.append(step_info)
        step_id += 1
        
    # this brings us to the data records
    # the pointer object holds the offset to the data records
    # use numpy from_file with this offset to efficiently extract data
    data = np.fromfile(fileName, dtype = BTS_RECORD_INFO, offset = ptr.getValue())
    dataframe = pd.DataFrame(data = data, columns = BTS_VARIABLE_LIST)
    # perform conversions on some of the columns into more recognizable units
    # change type of step_time to float
    dataframe["step_time"] = dataframe["step_time"].astype(float)
    # convert voltage from tenths of mV to V
    dataframe["voltage"] = TMV2V(dataframe["voltage"].astype(float))
    # convert current from uA to mA
    dataframe["current"] = UA2MA(dataframe["current"].astype(float))
    # convert capacity from uAs to mAh
    dataframe["Q-Q0"] = UAS2MAH(dataframe["Q-Q0"].astype(float))
    # convert energy from uWs to Wh
    dataframe["Energy"] = UWS2WH(dataframe["Energy"].astype(float))
    # wish there was a way to verify the checksum, but for now just delete it
    del dataframe["checksum"]
    return header_info, step_infos, dataframe


# testing
x1, x2, y = fromFile("../Other/_.nda")