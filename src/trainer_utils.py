from DFGCNN import DFGCNN
from mini_batch_loader import DatasetPreProcessor
from copy_model import copy_model

import os
import chainer
from chainer import config
from chainer import serializers
from chainer import cuda, Variable
from chainer import optimizers, serializers, training
from chainer.training import extensions


class EasyTrainer(object):
    def __init__(self):
        pass

    @staticmethod
    def prepare_model():
        model = DFGCNN(config.in_ch, config.n_class, config.M)
        if os.path.exists(config.initial_model):
            print('Load model from', config.initial_model, file=sys.stderr)
            serializers.load_npz(config.initial_model, model)

        if config.gpu >= 0:
            chainer.cuda.get_device(config.gpu).use()
            model.to_gpu()
        model.n_class = config.n_class
        m_eval = model.copy()

        return model, m_eval

    def select_optimizer(self):
        if config.training_params.optimizer=='RMSpropGraves':
            return chainer.optimizers.RMSpropGraves(config.training_params.lr)
        elif config.training_params.optimizer=='Adam':
            return chainer.optimizers.Adam()
        elif config.training_params.optimizer=='AdaDelta':
            return chainer.optimizers.AdaDelta()
        elif config.training_params.optimizer=='NesterovAG':
            return chainer.optimizers.NesterovAG(config.training_params.lr)
        elif config.training_params.optimizer=='MomentumSGD':
            return chainer.optimizers.MomentumSGD(config.training_params.lr)

    def prepare_optimizer(self, model):
        optimizer = self.select_optimizer()
        optimizer.setup(model)
        if config.training_params.weight_decay:
            optimizer.add_hook(chainer.optimizer.WeightDecay( \
                config.training_params.weight_decay))
        if config.training_params.lasso:
            optimizer.add_hook(chainer.optimizer.Lasso( \
                config.training_params.weight_decay))
        if config.training_params.clip_grad:
            optimizer.add_hook(chainer.optimizer.GradientClipping( \
                config.training_params.clip_value))
        return optimizer

    def prepare_dataset(self):
        # load dataset
        if chainer.config.data_set=='mnist':
            train_mini_batch_loader, test_mini_batch_loader = \
                chainer.datasets.get_mnist()
        else:
            train_mini_batch_loader = DatasetPreProcessor('train')
            test_mini_batch_loader = DatasetPreProcessor('test')

        if config.training_params.iter_type=='multi':
            iterator = chainer.iterators.MultiprocessIterator
        else:
            iterator = chainer.iterators.SerialIterator
        train_it = iterator( \
                        train_mini_batch_loader, \
                        config.training_params.batch_size, \
                        shuffle=config.train)

        val_batch_size = 1
        val_it = iterator( \
                    test_mini_batch_loader, \
                    val_batch_size, repeat=False, shuffle=False)
        return train_it, val_it, train_mini_batch_loader.__len__()

    def prepare_updater(self, train_it, optimizer):
        if config.training_params.updater_type=='standerd':
            return training.StandardUpdater( \
                train_it, optimizer, device=config.gpu)
        elif config.training_params.updater_type=='parallel':
            return training.ParallelUpdater( \
                train_it, optimizer, devices={'main': 1, 'second': 0})

    def run_trainer(self):
        # load model
        model, model_for_eval = self.prepare_model()
        print("---set model----------")

        # Setup optimizer
        optimizer = self.prepare_optimizer(model)
        print("---set optimzer----------")

        # load data
        train_it, val_it, train_data_length = self.prepare_dataset()
        print("---set data----------")

        updater = self.prepare_updater(train_it, optimizer)
        print("---set updater----------")

        evaluator_interval = config.training_params.report_epoch, 'epoch'
        snapshot_interval = config.training_params.snapshot_epoch, 'epoch'
        log_interval = config.training_params.report_epoch, 'epoch'

        trainer = training.Trainer( updater, \
            (config.training_params.epoch, 'epoch'), out=config.output_path)
        trainer.extend( \
            extensions.Evaluator(val_it, model_for_eval, device=config.gpu), \
            trigger=evaluator_interval)
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
        trainer.extend(extensions.snapshot_object( \
            model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
        if config.training_params.optimizer!='Adam' \
                    and config.training_params.optimizer!='AdaDelta':
            trainer.extend(extensions.ExponentialShift( \
                'lr', config.training_params.decay_factor), \
                trigger=(config.training_params.decay_epoch, 'epoch'))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport([ \
            'epoch', 'iteration', 'main/loss', 'validation/main/loss', \
            'main/accuracy', 'validation/main/accuracy', \
            ]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=1))
        print("---set trainer----------")

        if os.path.exists(config.resume):
            print('resume trainer:{}'.format(config.resume))
            # Resume from a snapshot
            serializers.load_npz(config.resume, trainer)

        trainer.run()
