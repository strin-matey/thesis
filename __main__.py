import torch.nn as nn
import torch.optim as optim
import time
import random as rm
# import matplotlib.pyplot as plt
import numpy as np
from src.utils import *

args = parse_arguments()

set_seed(args.seed)

# Load train and test datasets
trainloader, testloader = load_dataset(dataset_name=args.dataset, minibatch=args.batch_size,
                                       numworkers=args.numworkers, cluster=args.cluster)

# Load network
model = load_net(net=args.net, dataset_name=args.dataset)
if args.gpu:
    model = model.cuda()

# Preprocessing functional to a specific method
# Three possible training methods: Minibatch Persistency, Adaptive Nesterov and CLR
# The cycle_length of the CLR method is entered manually. In the paper are given proper bounds (2-10 times the number of iterations)
criterion = nn.CrossEntropyLoss()
optimizer = choose_optimizer(args.optimizer, model, args)

val_loss, val_acc, train_loss, train_acc, train_loss_sample, train_acc_sample, times = [['Nothing' for i in range(args.epochs)] for j in range(7)]

#train_loss_sample, train_acc_sample = [[['Nothing' for i in range(args.epochs)] for k in range(10)] for j in range(2)]

plot = []

percentage = 20

epoch_counter = 0

start = time.clock()

for epoch in range(args.epochs):
    print("Epoch: ", epoch)

    train(trainloader, model, optimizer, criterion, 1, gpu=args.gpu)

    val_loss[epoch], val_acc[epoch] = test(testloader, model)

    train_loss[epoch], train_acc[epoch] = test_train(trainloader, model)

    #train_loss_sample[epoch], train_acc_sample[epoch] = test_train_sample(trainloader, model, n_minibatches=10)

    '''for i in range(10):
        print("Stimato su n = ", i + 1)

       
        print("True value: ", train_loss[epoch], ", estimated value: ", train_loss_sample[i][epoch])
        print("True value acc: ", train_acc[epoch], ", estimated value acc: ", train_acc_sample[i][epoch])
    '''

    times[epoch] = time.clock() - start


del model

np.savez(f'results_dataaum_{args.optimizer}_{args.dataset}_{args.net}_{args.batch_size}_{args.epochs}_lr{args.lr}_patience{args.patience}',
         times=times,
         validation_accuracy=val_acc,
         validation_loss=val_loss,
         train_accuracy=train_acc,
         train_loss=train_loss,
         train_accuracy_sample=train_acc_sample,
         train_loss_sample=train_loss_sample,
         plot=plot,
         epochs=np.array([args.epochs]),  # TODO inserire job-ID
         lr=np.array([args.lr]),
         momentum=np.array([args.momentum]))
