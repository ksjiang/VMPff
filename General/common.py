# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 18:24:01 2023

@author: Kyle
"""

# imports
import os
import struct
from matplotlib.patches import Rectangle, Wedge
import numpy as np
import scipy.integrate

# sizes of data types
BYTE_SIZE = 1
SHORT_SIZE = 2
LONG_SIZE = 4
LLONG_SIZE = 8

# data types and their struct symbolcodes
SINGLE = ("<f", LONG_SIZE)
DOUBLE = ("<d", LLONG_SIZE)
UINT8 = ("<B", BYTE_SIZE)
UINT16 = ("<H", SHORT_SIZE)
UINT32 = ("<I", LONG_SIZE)

# big-endian (gross!)
SINGLE_BE = (">f", LONG_SIZE)
DOUBLE_BE = (">d", LLONG_SIZE)

# to denote an empty byte or bit field
NONE_FIELD = (None, None)

# a pointer object that can be used to keep track of position within a data block
class pointer(object):
    def __init__(self, value = 0):
        self.value = value
        return
    
    def getValue(self):
        return self.value
    
    def setValue(self, newValue):
        self.value = newValue
        return
    
    def add(self, delta):
        self.setValue(self.getValue() + delta)
        return
    
def checkField(data, ptr, expected):
    assert data[ptr.getValue(): ptr.getValue() + len(expected)] == expected
    ptr.add(len(expected))
    return

def checkFieldFlat(data, ptrVal, expected):
    assert data[ptrVal: ptrVal + len(expected)] == expected
    return

def getRaw(data, ptr, size):
    r = data[ptr.getValue(): ptr.getValue() + size]
    ptr.add(size)
    return r

def getRawFlat(data, ptrVal, size):
    return data[ptrVal: ptrVal + size]

# returns a field value and updates pointer object
def getField(data, ptr, fmt, size):
    r = struct.unpack(fmt, data[ptr.getValue(): ptr.getValue() + size])[0]
    # update pointer value
    ptr.add(size)
    return r

# returns a field value with a static offset
def getFieldFlat(data, ptrVal, fmt, size):
    return struct.unpack(fmt, data[ptrVal: ptrVal + size])[0]

def getFieldsStridedFlat(data, ptrVal, blk, N, fmt, size):
    r = [0] * N
    for idx in range(N):
        r[idx] = getFieldFlat(data, ptrVal + idx * blk, fmt, size)
        
    return r

def getBitField(bt, bitPos, fieldSize):
    return (bt >> bitPos) % (1 << fieldSize)

def getBitFieldFlat(data, ptrVal, bitPos, fieldSize):
    return getBitField(getFieldFlat(data, ptrVal, *UINT8), bitPos, fieldSize)

def getBitFieldsStridedFlat(data, ptrVal, blk, N, bitPos, fieldSize):
    r = [0] * N
    for idx in range(N):
        r[idx] = getBitFieldFlat(data, ptrVal + idx * blk, bitPos, fieldSize)
        
    return r

# versions of the above functions for multiprocessing
def initProcess(sharedData):
    global mySharedData
    mySharedData = sharedData
    return

def getFieldFlatShared(ptrVal, fmt, size):
    global mySharedData
    return getFieldFlat(mySharedData, ptrVal, fmt, size)

def getFieldsStridedFlatShared(ptrVal, blk, N, fmt, size):
    global mySharedData
    return getFieldsStridedFlat(mySharedData, ptrVal, blk, N, fmt, size)

def getBitFieldFlatShared(ptrVal, bitPos, fieldSize):
    global mySharedData
    return getBitFieldFlat(mySharedData, ptrVal, bitPos, fieldSize)

def getBitFieldsStridedFlatShared(ptrVal, blk, N, bitPos, fieldSize):
    global mySharedData
    return getBitFieldsStridedFlat(mySharedData, ptrVal, blk, N, bitPos, fieldSize)

def ternary(condition, valueIfTrue, valueIfFalse):
    if condition:
        return valueIfTrue
    else:
        return valueIfFalse

# floating-point "not a number"
NAN = float("NaN")
EPS = np.finfo(float).eps

# some styles that we can use to keep consistency across figures

# output image settings
IMAGE_DPI = 500
STANDARD_DIMS = (6, 4)
SMALL_DIMS = (5, 4)
WIDE_DIMS = (7, 4)
TALL_DIMS = (4, 6)
CAPTION_SIZE = 16
LONG_CAPTION_SIZE = 12
CBAR_PAD = 20
ROT_FACE_LEFT = 270
ROT_LABEL_DOWNRIGHT = -45
MARKER_SIZE = 9
MARKER_WIDTH_HOLLOW = 2
MARKER_DISK = 'o'
MARKER_CROSS = '+'
ERRORBAR_SIZE = 5
GREY_PATCH = (0.5, 0.5, 0.5, 0.5)
MARKER_WIDTH_SERIES_LIMS = (1, 3)
GREYSCALE_SERIES_LIMS= (0.7, 0.2)

# font settings
fontSettings = {
        "family": "Arial", 
        "weight": "normal", 
        "size": CAPTION_SIZE, 
        }

captionFont = {
        "size": 14, 
        }

legendFont = {
        "size": 10,
        }

def intersperse(l, item):
    r = [item] * (2 * len(l) - 1)
    r[: : 2] = l
    return r

# collect all files in a directory that have a substring
# by default, get *all* files in directory
def collectFiles(inputFileDir, s = ""):
    return [fname for fname in os.listdir(inputFileDir) if os.path.isfile(os.path.join(inputFileDir, fname)) and s in fname]

def createOutputFileName(inputFiles, suffix, extension = None):
    if len(inputFiles) == 0: assert False
    r, ext = os.path.splitext(inputFiles[0])
    roots = [r]
    # strip extensions
    for inputFile in inputFiles[1: ]:
        r, ext_ = os.path.splitext(inputFile)
        if ext != ext_: assert False
        roots.append(r)
        
    common_part = os.path.commonprefix(roots)
    if extension is not None:
        ext = '.' + extension
        
    return '_'.join([common_part, suffix]) + ext

def printElapsedTime(tag, start, finish):
    print("%s finished in %0.6f s" % (tag, finish - start))
    return

PATCH_WIDTH = 0.5
PATCH_PAD = 0.7

def drawErrorBounds(ax, xpos, bottomcent, topcent, width = PATCH_WIDTH, patch_color = GREY_PATCH):
    # draw a rectangle
    ax.add_patch(Rectangle((xpos - width / 2, bottomcent), width, topcent - bottomcent, fill = True, color = patch_color, linewidth = 0))
    
    # draw semicircles at both ends
    ax.add_patch(Wedge((xpos, topcent), width / 2, 0, 180, fill = True, color = patch_color, linewidth = 0))
    ax.add_patch(Wedge((xpos, bottomcent), width / 2, 180, 360, fill = True, color = patch_color, linewidth = 0))
    return

def Integrate(y, x, xleft, xright):
    mask = (x > xleft) & (x < xright)
    return scipy.integrate.simps(y[mask], x[mask])

def IntegrateWithBaseline(y, x, xleft, xright):
    yleft, yright = np.interp(xleft, x, y), np.interp(xright, x, y)
    below_area = 0.5 * (xright - xleft) * (yleft + yright)
    total_area = Integrate(y, x, xleft, xright)
    return total_area - below_area

def createTicks(left_bound, right_bound, step):
    # leave an epsilon of room on either side when calculating tick positions
    left_bound = left_bound * (1 - EPS)
    right_bound = right_bound * (1 + EPS)
    # the leftmost tick must be right of (or equal to) left_bound
    true_left = (left_bound // step) * step
    if (left_bound - true_left) / step > EPS:
        true_left += step
        
    # the rightmost tick must be left of (or equal to) right_bound
    true_right = (right_bound // step) * step
    if (right_bound - true_right) / step < -EPS:
        true_right -= step
        
    return np.arange(true_left, true_right + 0.5 * step, step)