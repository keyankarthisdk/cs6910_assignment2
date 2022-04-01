'''
Questions Part A
'''

# Imports
import json
import functools
from pprint import pprint

from Model import *

# Main Functions
# Wandb Sweep Function for Part A @ Karthikeyan S CS21M028
def Model_Sweep_Run(wandb_data):
    # Init
    wandb.init()

    # Get Run Config
    config = wandb.config
    N_EPOCHS = config.n_epochs
    BATCH_SIZE = config.batch_size

    N_FILTERS = config.n_filters
    FILTER_SIZE = config.filter_size
    DROPOUT = config.dropout
    BATCH_NORM = config.batch_norm
    DENSE_NEURONS = config.dense_neurons
    DENSE_DROPOUT = config.dense_dropout

    LEARNING_RATE = config.lr

    print("RUN CONFIG:")
    pprint(config)

    # Get Inputs
    N_FILTERS_LAYERWISE = [int(N_FILTERS[i]) for i in range(5)]
    inputs = {
        "img_size": (227, 227, 3), 
        "Y_shape": len(DATASET_INATURALIST_CLASSES), 
        "model": {
            "blocks": [
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[0], conv_kernel_size=(FILTER_SIZE[0], FILTER_SIZE[0]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[1], conv_kernel_size=(FILTER_SIZE[1], FILTER_SIZE[1]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[2], conv_kernel_size=(FILTER_SIZE[2], FILTER_SIZE[2]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[3], conv_kernel_size=(FILTER_SIZE[3], FILTER_SIZE[3]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[4], conv_kernel_size=(FILTER_SIZE[4], FILTER_SIZE[4]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
            ], 
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
        shuffle=True, data_aug=True
    )
    inputs["dataset"] = DATASET

    # Build Model
    MODEL = Model_SequentialBlocks(
        X_shape=inputs["img_size"], Y_shape=inputs["Y_shape"], 
        Blocks=inputs["model"]["blocks"],
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
    "name": "part-a-run-1",
    "method": "grid",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "n_epochs": {
            "values": [10]
        },
        "batch_size": {
            "values": [128]
        },

        "filter_size": {
            "values": [
                [3, 3, 3, 5, 7]
            ]
        },
        "n_filters": {
            "values": [
                [32, 64, 64, 128, 128]
            ]
        },
        "dropout": {
            "values": [0.1]
        },
        "batch_norm": {
            "values": [True]
        },

        "dense_neurons": {
            "values": [512]
        },
        "dense_dropout": {
            "values": [0.1]
        },

        "lr": {
            "values": [0.001]
        }
    }
}

# Run Sweep
sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
# sweep_id = ""
TRAINER_FUNC = functools.partial(Model_Sweep_Run, wandb_data=WANDB_DATA)
wandb.agent(sweep_id, TRAINER_FUNC, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], count=1)