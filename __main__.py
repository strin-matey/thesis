import torch.nn as nn
import torch.optim as optim
import time
# import matplotlib.pyplot as plt
from src.utils import *

args = parse_arguments()

set_seed(args.seed)

# Load train and test datasets
trainloader, testloader = load_dataset(dataset_name=args.dataset, minibatch=args.batch_size,
                                       numworkers=args.numworkers, cluster=args.cluster)

# Load network
model = load_net(net=args.net, dataset_name=args.dataset).cuda()

# Preprocessing functional to a specific method
# Three possible training methods: Minibatch Persistency, Adaptive Nesterov and CLR
# The cycle_length of the CLR method is entered manually. In the paper are given proper bounds (2-10 times the number of iterations)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)


for epoch in range(args.epochs):  # loop over the dataset multiple times

    start = time.time()
    print("epoch: ", epoch)

    train(trainloader, model, optimizer, criterion, 1)

    end = time.time()

    print('time: ', end - start)

    print("train: ", test_train(trainloader, model))

    print("test: ", test(testloader, model))

del model


