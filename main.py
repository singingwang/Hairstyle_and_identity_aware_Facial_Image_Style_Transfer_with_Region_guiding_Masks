import time
import numpy as np
from RAGAN import *

def main():
    gan = RAGAN()
    gan.define_models()
    gan.define_training_models()
    gan.train(epoches=30, start_epoch=0)

if __name__ == '__main__':
    #gpu_options = tf.GPUOptions(allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    #設定 Keras 使用的 Session
    #tf.keras.backend.set_session(sess)
    main()