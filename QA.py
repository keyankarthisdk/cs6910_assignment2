'''
Questions Part A
'''

# Imports
import json

from Model import *

# Main Functions
# Wandb Sweep Function for Part A @ Karthikeyan S CS21M028


# Run
# Params
WANDB_DATA = json.load(open("config.json", "r"))
WANDB_DATA.update({
    "use_wandb": True
})
# Params

# Run