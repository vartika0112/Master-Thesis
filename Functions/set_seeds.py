import os
import random
import numpy as np
import tensorflow as tf

def reset_random_seeds():
    SEED = 123
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

