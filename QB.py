'''
Questions Part B
'''

# Imports
import json

from Model import *

# Main Functions
# Wandb Sweep Function for Part B @ N Kausik CS21M037


# Run
# Params
WANDB_DATA = json.load(open("config.json", "r"))
WANDB_DATA.update({
    "use_wandb": True
})
# Params

# Run