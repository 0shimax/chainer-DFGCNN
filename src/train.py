import args_settings
from trainer_utils import EasyTrainer
import os
from chainer import global_config, config


if __name__ == '__main__':
    print("-------traing")
    global_config.data_set = 'mnist'
    global_config.n_class = 10
    global_config.in_ch = 1
    global_config.gpu = -1
    global_config.output_path = os.path.join(config.data_root_path, 'results')
    global_config.initial_model = os.path.join(config.output_path, 'None')
    global_config.resume = os.path.join(config.output_path, 'None')

    # global_config.debug = True
    e_trainer = EasyTrainer()
    e_trainer.run_trainer()
