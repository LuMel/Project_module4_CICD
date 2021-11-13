import os
import uvicorn
from src.train_data import split_and_train
from src import fastAPI_main


if __name__ == "__main__":
    uvicorn.run("src.fastAPI_main:app")
    #split_and_train()
