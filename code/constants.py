import torch

TRAIN_SIZE = 20
PREDICT_SIZE  = 5
BATCH_SIZE = 16
LSTM_NUM_LAYERS  = 3
HIDDEN_SIZE = 64
INPUT_SIZE = 880
IMAGE_SHAPE = (24,64)
MAX_EPOCHS = 25
INITIAL_LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")