import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

filename = 'Resnet34/results_dataaum_cifar10_resnet34_512_300_lr0.1_patience0'
filename2 = 'Resnet34/results_dataaum_cifar10_resnet34_512_300_lr0.1_patience15'

with np.load(f'{filename}.npz') as fl:
    with np.load(f'{filename2}.npz') as fl2:
        fig, ax = plt.subplots()

        #res = []
        #for epoch in range(5, 300):
        #    if fl['validation_loss'][epoch] - fl['validation_loss'][epoch - 15] > 0:
        #        res.append(True)
            #else:
            #    res.append(False)

        #print(len(res))
        var = fl['train_loss']
        print("Minimum standard ", np.min(var))
        #print("Maximum standard ", np.max(var), " at epoch: ", np.argwhere(var == np.max(var)))
        #max_standard = np.min(var)
        #print(max_standard)

        var2 = fl2['train_loss']
        #var2 = var2[199:299]
        #max_cyclic = np.argwhere(var2 == np.min(var2))
        print("Minimum cyclic: ", np.min(var2), " at epoch: ", np.argwhere(var2 == np.min(var2)))
        #print("Maximum cyclic: ", np.max(var2))

        #print(max_cyclic)
        ax.plot(np.arange(0, len(var)), var, '--', label='Standard')
        ax.plot(np.arange(0, len(var2)), var2, '--', label='Cyclic')

        ax.set(xlabel='Epoch', ylabel='Loss',
               title='Resnet34 on CIFAR 10 (training loss)')

        ax.legend()

        plt.show()

        fig.savefig(f'{filename}_training_loss.pdf')


