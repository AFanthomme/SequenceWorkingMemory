'''This module helps loading / saving data'''
import json
import os

from environment import *
from nets import *

class EnvEncDec:

    def __init__(self, filename):
        self.env_register = {
            'DiscreteDots': DiscreteDots,
            'ContinuousDots': ContinuousDots,
        }

        self.enc_register = {
            'TensorSequenceEncoder': TensorSequenceEncoder,
            'RNNSequenceEncoder': RNNSequenceEncoder,
        }

        self.get_env_enc_dec(filename)


    def get_env_enc_dec(self, filename):
        with open(filename, 'r') as f:
            full_params = json.load(f)

        self.full_params = full_params
        enc_params = full_params['enc_params']
        env_params = full_params['env_params']
        dec_params = full_params['dec_params']

        enc_name = full_params['enc_name']
        env_name = full_params['env_name']
        dec_name = full_params['dec_name']


    def store_env_enc_dec(self, filename)