from mini_batch_loader import DatasetPreProcessor
import args_settings
from trainer_utils import EasyTrainer

import chainer
from chainer import serializers, Variable
from chainer import config, global_config

import os
import numpy as np
from collections import defaultdict


def test(model):
    sum_accuracy = 0
    sum_loss     = 0
    omit_counter = 0
    heat_map = defaultdict(dict)

    val_it, test_data_size, pairs = prepare_dataset()
    print("------test data size")
    print(test_data_size)

    out_file = os.path.join(config.output_path, config.output_file_name)
    with open(out_file, 'w') as of:
        s = ','+','.join(map(str, config.label_exist))
        of.write(s+'\n')

        for gt_label in config.label_exist:
            out_list = []
            out_list.append(gt_label)
            for infer_label in config.label_exist:
                cnt = heat_map[gt_label].get(infer_label,0)
                out_list.append(cnt)
            s = ','.join(map(str, out_list))
            of.write(s+'\n')

    print("test mean accuracy {}".format(sum_accuracy/(test_data_size-omit_counter)))


def prepare_dataset():
    # load dataset
    test_mini_batch_loader = DatasetPreProcessor('test')
    val_it = chainer.iterators.SerialIterator( \
                            test_mini_batch_loader, \
                            1, repeat=False, shuffle=False)
    return val_it, test_mini_batch_loader.__len__(), test_mini_batch_loader.pairs


def main(args):
    _, model_eval = EasyTrainer.prepare_model()
    test(model_eval)


if __name__ == '__main__':
    global_config.train = False
    config.label_exist = list(range(10))
    config.output_file_name = 'result_heat_map.csv'
    main()
