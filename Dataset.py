'''
Dataset
'''

# Imports
import os
import shutil
import numpy as np
np.random.seed(0)
from tqdm import tqdm
from keras.preprocessing import *
from keras.preprocessing.image import *


# Main Vars
DATASET_PATH_INATURALIST = "Dataset/inaturalist_12K"
DATASET_INATURALIST_CLASSES = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

# Main Functions
# Load Train and Test Dataset Functions @ Karthikeyan S CS21M028

# Get Random Image Path from Dataset @ N Kausik CS21M037
def GetImagePath_Random():
    '''
    Get Random Image Path in train dataset
    '''
    dataset_path = os.path.join(DATASET_PATH_INATURALIST, "train")
    class_random = os.listdir(dataset_path)[np.random.randint(0, 10)]
    class_Is = os.listdir(os.path.join(dataset_path, class_random))
    I_name = class_Is[np.random.randint(0, len(class_Is))]
    I_path = os.path.join(dataset_path, class_random, I_name)
    return I_path, class_random

# Run