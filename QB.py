'''
Questions Part B
'''

# Imports
import json
import functools
from pprint import pprint

from Model import *

# Main Functions
# Wandb Sweep Function for Part B @ N Kausik CS21M037
def Model_Sweep_Run(wandb_data):
    # Init
    wandb.init()

    # Get Run Config
    config = wandb.config
    MODEL_NAME = config.model_name
    N_EPOCHS = config.n_epochs
    BATCH_SIZE = config.batch_size
    DATA_AUG = config.data_aug
    UNFREEZE_COUNT = config.unfreeze_count

    DENSE_NEURONS = config.dense_neurons
    DENSE_DROPOUT = config.dense_dropout

    LEARNING_RATE = config.lr

    print("RUN CONFIG:")
    pprint(config)

    # Get Inputs
    inputs = {
        "img_size": (224, 224, 3), 
        "Y_shape": len(DATASET_INATURALIST_CLASSES), 
        "model": {
            "compile_params": {
                "loss_fn": "categorical_crossentropy",
                "optimizer": Adam(learning_rate=LEARNING_RATE),
                "metrics": ["accuracy"]
            }
        }
    }

    # Get Train Val Dataset
    DATASET = LoadTrainDataset_INaturalist(
        DATASET_PATH_INATURALIST,  
        img_size=inputs["img_size"][:2], batch_size=BATCH_SIZE, 
        shuffle=True, data_aug=DATA_AUG
    )
    inputs["dataset"] = DATASET

    # Build Model
    MODEL = Model_PretrainedBlocks(
        X_shape=inputs["img_size"], Y_shape=inputs["Y_shape"], 
        model_name=MODEL_NAME,
        unfreeze_count=UNFREEZE_COUNT,
        dense_n_neurons=DENSE_NEURONS, dense_dropout_rate=DENSE_DROPOUT
    )
    MODEL = Model_Compile(MODEL, **inputs["model"]["compile_params"])

    # Train Model
    TRAINED_MODEL, TRAIN_HISTORY = Model_Train(MODEL, inputs, N_EPOCHS, wandb_data)

    # Load Best Model
    TRAINED_MODEL = Model_LoadModel("Models/best_model.h5")
    # Get Test Dataset
    DATASET_PATH_INATURALIST_TEST = os.path.join(DATASET_PATH_INATURALIST, "val")
    DATASET_TEST = LoadTestDataset_INaturalist(
        DATASET_PATH_INATURALIST_TEST, 
        img_size=inputs["img_size"][:2], batch_size=BATCH_SIZE, 
        shuffle=False
    )
    # Test Best Model
    loss_test, eval_test = Model_Test(TRAINED_MODEL, DATASET_TEST)

    # Wandb log test data
    wandb.log({
        "loss_test": loss_test,
        "eval_test": eval_test
    })

    # Close Wandb Run
    # run_name = "ep:"+str(N_EPOCHS) + "_" + "bs:"+str(BATCH_SIZE) + "_" + "nf:"+str(N_FILTERS) + "_" + str(DROPOUT)
    # wandb.run.name = run_name
    wandb.finish()

# Run
# Params
WANDB_DATA = json.load(open("config.json", "r"))
# Params

# Run
# Sweep Setup
SWEEP_CONFIG = {
    "name": "part-b-run-1",
    "method": "grid",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "model_name":{
            "values": ["Xception"]
        },

        "n_epochs": {
            "values": [10]
        },
        "batch_size": {
            "values": [512]
        },
        "data_aug":{
            "values": [True]
        },

        "unfreeze_count":{
            "values": [20]
        },

        "dense_neurons": {
            "values": [512]
        },
        "dense_dropout": {
            "values": [0.1]
        },

        "lr": {
            "values": [0.0001]
        }
    }
}

# Run Sweep
sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
# sweep_id = ""
TRAINER_FUNC = functools.partial(Model_Sweep_Run, wandb_data=WANDB_DATA)
wandb.agent(sweep_id, TRAINER_FUNC, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], count=1)