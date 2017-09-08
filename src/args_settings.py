import chainer
from chainer import global_config
import os
from easydict import EasyDict as edict
from math import sqrt


# immortal params
data_root_path = './data'
global_config.data_root_path = data_root_path
global_config.M = 9  # filter weight size= w*h
# global_config.n_class = 10
# global_config.in_ch = 1

augmentation_params = {
                       'scale':[0.75, 0.875, 1.125, 1.25],  # [0.5, 0.75, 1.125, 1.25]
                       'ratio':[sqrt(1/2), 1, sqrt(2)],
                       'lr_shift':[-64, -32, -16, 16, 32, 64],
                       'ud_shift':[-64, -32, -16, 16, 32, 64],
                       'rotation_angle': list(range(5,360,5))
                      }

dic_name = 'sysm_pathological.dict'
token_args = {
        'lang': 'ja',
        'max_len': 50,
        'tagger': '-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati',
        'dic_load_path': os.path.join(data_root_path, dic_name),
        'texts_path': os.path.join(data_root_path, 'pathological_comments'),
        'dic_save_path': os.path.join(data_root_path, dic_name),
        'tokens_path': os.path.join(data_root_path, 'sysm_tokens.npz')
    }

training_params = {
        'optimizer': 'NesterovAG',
        'lr': 1e-3,
        'batch_size': 20,
        'epoch': 100,
        'decay_factor': 0.1,  # as lr time decay
        'decay_epoch': 50,
        'snapshot_epoch': 20,
        'report_epoch': 1,
        'weight_decay': True,
        'lasso': False,
        'clip_grad': False,
        'weight_decay': 0.0005,
        'clip_value': False,  # 5.,
        'iter_type': 'serial',
        'updater_type': 'standerd',
    }


global_config.augmentation_params = edict(augmentation_params)
global_config.training_params = edict(training_params)
global_config.token_args = edict(token_args)
