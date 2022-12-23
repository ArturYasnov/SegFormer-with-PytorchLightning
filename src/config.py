import os


class CFG:
    BASE_DIR = os.getcwd()
    MODELS_DIR = f"{BASE_DIR}/models"
    LOGS_DIR = f"{BASE_DIR}/logs"
    DATA_DIR = f"{BASE_DIR}/data"


class Train_CFG:
    experiment_name = "base_train"
    image_size = 512

    epocs = 100
    lr = 6e-5 * 8
    train_bs = 4
    valid_bs = 4
    accumulate_bs = 8

    # scheduler
    step = [20, 45, 90]
    gamma = 0.2
