import setka
import setka.base
import setka.callbacks

import torch

import tiny_model
import test_dataset
from test_metrics import tensor_loss as loss

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
                                            lr=0.1,
                                            momentum=0.9,
                                            weight_decay=5e-4)
                                    ]
                                 ),
                                 setka.callbacks.ProgressBar()
                             ])

trainer.one_epoch('train', 'train')
trainer.one_epoch('valid', 'train')
