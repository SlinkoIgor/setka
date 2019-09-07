import setka
import setka.base
import setka.callbacks

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

from test_metrics import tensor_loss as loss
from test_metrics import tensor_acc as acc

ds = test_dataset.CIFAR10()
model = tiny_model.TensorNet()

trainer = setka.base.Trainer(callbacks=[
                                 setka.callbacks.DataSetHandler(ds, batch_size=32, limits=2),
                                 setka.callbacks.ModelHandler(model),
                                 setka.callbacks.LossHandler(loss),
                                 setka.callbacks.OneStepOptimizers(
                                    [
                                        setka.base.OptimizerSwitch(
                                            model,
                                            torch.optim.SGD,
                                            lr=0.01,
                                            momentum=0.9,
                                            weight_decay=5e-4)
                                    ]
                                 ),
                                 setka.callbacks.ComputeMetrics([loss, acc]),
                                 setka.callbacks.MakeCheckpoints('acc', max_mode=True, name='acc'),
                                 setka.callbacks.MakeCheckpoints('loss', max_mode=False, name='loss')
                             ])

for index in range(10):
    trainer.one_epoch('train', 'train')
    trainer.one_epoch('valid', 'train')
    trainer.one_epoch('valid', 'valid')