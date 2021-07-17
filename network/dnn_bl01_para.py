import numpy as np
import os

class dnn_bl01_para(object):
    """
    define a class to store parameters
    """

    def __init__(self):

        #=======Layer 01: full connection
        self.l01_fc             = 2048 
        self.l01_is_act         = True
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = False
        self.l01_drop_prob      = 0.2

        #=======Layer 02: full connection
        self.l02_fc             = 2048
        self.l02_is_act         = True
        self.l02_act_func       = 'RELU'
        self.l02_is_drop        = False
        self.l02_drop_prob      = 0.2

        #=======Layer 03: full connection
        self.l03_fc             = 512
        self.l03_is_act         = True
        self.l03_act_func       = 'RELU'
        self.l03_is_drop        = False
        self.l03_drop_prob      = 0.2

        #=======Layer 04: Final layer
        self.l04_fc             = 10   
        self.l04_is_act         = False
        self.l04_act_func       = 'RELU'
        self.l04_is_drop        = False
        self.l04_drop_prob      = 1

