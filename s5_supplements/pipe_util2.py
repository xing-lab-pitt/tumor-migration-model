import glob
import os 
import shutil
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from itertools import chain
from datetime import datetime
from skimage import filters
import pickle


def folder_verify(folder):
    """Verify if the folder string is ended with '/' """
    if folder[-1]!="/":
        folder = folder+"/"
    return folder


def folder_file_num(folder, pattern = "*", if_verbose = 0):
    """How many files in the folder"""
    if folder[-1]!="/":
        folder = folder +"/"
    file_list = sorted(glob.glob(folder + "*" + pattern + "*"))
    if if_verbose:
        print("%s "%folder + "has %s files"%len(file_list))
    return(file_list)


def create_folder(folder):
    """Create a folder. If the folder exist, erase and re-create."""
    folder = folder_verify(folder)
    
    if os.path.exists(folder): # recreate folder every time. 
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)
    print("%s folder is freshly created. \n"%folder)
    
    return folder
    