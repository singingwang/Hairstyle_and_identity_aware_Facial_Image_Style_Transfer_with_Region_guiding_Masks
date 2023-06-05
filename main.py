import time
import numpy as np
from RAGAN import *

def main():
    gan = RAGAN()
    gan.define_models()
    gan.define_training_models()
    gan.train(epoches=30, start_epoch=0)

if __name__ == '__main__':
    main()
