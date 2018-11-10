import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import math
import argparse
import os
import random as rm

from src.resnet import *
from src.alexnet import *
from src.vgg import *


def parse_arguments():
    # Parse the imput parameters
    parser = argparse.ArgumentParser(description='Cifar testing')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--update-number', type=int, default=1, metavar='N',
                        help='number of gradient update per minibatch (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--patience', type=int, default=0, metavar='P',
                        help='Early-stopping restart')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--numworkers', type=int, default=1, metavar='S',
                        help='numworkers for the loading class (default: 1)')
    parser.add_argument('--net', dest="net", action="store", default="vgg16",
                        help='network architecture to be used')
    parser.add_argument('--dataset', dest="dataset", action="store", default="cifar10",
                        help='dataset on which to train the network')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--outfolder', type=str,
                        help='Out folder for results')
    parser.add_argument('--optimizer', type=str,
                        help='Optimizer to use')
    parser.add_argument('--clr', dest='clr', action='store_true')
    parser.add_argument('--no-clr', dest='clr', action='store_false')
    parser.set_defaults(clr=False)
    parser.add_argument('--alt-train', dest='alt', action='store_true')
    parser.add_argument('--no-alt-train', dest='alt', action='store_false')
    parser.set_defaults(alt=False)
    parser.add_argument('--cluster', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)

    return parser.parse_args()


def set_seed(seed):
    # Reproducible runs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def choose_optimizer(opt, model, args):

    if opt == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False)
    elif opt == 'nesterov':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters())
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif opt == 'adamax':
        optimizer = optim.Adamax(model.parameters())
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())

    return optimizer


def my_adaptive_nesterov(trainloader, model, optimizer, criterion):
    v_t = []

    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs = inputs
        labels = labels
        if i == 0:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for k, f in enumerate(model.parameters(), 0):
                v_t.append(0.1 * f.grad.data.clone())
        else:
            for k, f in enumerate(model.parameters()):
                f.data.sub_(0.1 * v_t[k])

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            g = torch.tensor([0], dtype=torch.float32)

            for k, f in enumerate(model.parameters(), 0):
                g = torch.cat((g, f.grad.data.clone().view(-1)), dim=-1)

            print(g.size())

            epsilon = torch.clamp(loss.clone() / torch.clamp(torch.dot(g, g), max=1e10).data, min=0.001, max=0.1)

            print(epsilon.data)
            print(epsilon.data.size())

            # d = torch.distributions.normal.Normal(torch.Tensor([0.0]), torch.Tensor([torch.std(g)]))

            for k, f in enumerate(model.parameters(), 0):
                # if i%5 == 0:
                # f.data.sub_(f.grad.data.add_(d.sample().cuda()) * epsilon.data)
                # else:
                f.data.sub_(f.grad.data * epsilon.data)
                v_t[k] = 0.5 * v_t[k] + f.grad.data * epsilon.data

        del inputs, labels, outputs


# Minibatch Persistency, with update_number = 1 corresponds to the standard method
def train(trainloader, model, optimizer, criterion, update_number, gpu=True):
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        for j in range(update_number):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        del inputs, labels, outputs


def load_dataset(dataset_name="cifar10", minibatch=64, numworkers=2, cluster=False):

    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 	#uncomment for data augmentation
            transforms.RandomHorizontalFlip(),      #uncomment for data augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        trainset = torchvision.datasets.CIFAR10(root='/home/met/PycharmProjects/thesis/data' if not cluster else '/home/stringherm/thesis/data', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch,
                                                  shuffle=True, num_workers=numworkers)
        testset = torchvision.datasets.CIFAR10(root='/home/met/PycharmProjects/thesis/data' if not cluster else '/home/stringherm/thesis/data', train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch,
                                                 shuffle=True, num_workers=numworkers)
        return trainloader, testloader
    elif dataset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))])
        trainset = torchvision.datasets.CIFAR100(root='/home/met/PycharmProjects/thesis/data' if not cluster else '/home/stringherm/thesis/data',
                                                 train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch,
                                                  shuffle=True, num_workers=numworkers)
        testset = torchvision.datasets.CIFAR100(root='/home/met/PycharmProjects/thesis/data' if not cluster else '/home/stringherm/thesis/data',
                                                train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch,
                                                 shuffle=True, num_workers=numworkers)
        return trainloader, testloader
    elif dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch,
                                                  shuffle=True, num_workers=numworkers)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch,
                                                 shuffle=True, num_workers=numworkers)
        return trainloader, testloader


def load_net(net="alexnet", dataset_name="cifar10"):
    num_classes = 10
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    elif dataset_name == "imagenet":
        num_classes = 1000

    if net == "alexnet":
        return alexnet(num_classes)
    elif net == "resnet18":
        return ResNet18()
    elif net == "resnet34":
        return ResNet34()
    elif net == "resnet50":
        return ResNet50()
    elif net == "resnet101":
        return ResNet101()
    elif net == "vgg11":
        return VGG("VGG11", num_classes)
    elif net == "vgg16":
        return VGG("VGG16", num_classes)
    elif net == "vgg19":
        return VGG("VGG19", num_classes)


def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
    """implements a cyclical learning rate policy (CLR).
    Notes: the learning rate of optimizer should be 1

    Parameters:
    ----------
    mode : str, optional
        one of {triangular, triangular2, exp_range}.
    scale_md : str, optional
        {'cycles', 'iterations'}.
    gamma : float, optional
        constant in 'exp_range' scaling function: gamma**(cycle iterations)

    Examples:
    --------
    >>> # the learning rate of optimizer should be 1
    >>> optimizer = optim.SGD(model.parameters(), lr=1.)
    >>> step_size = 2*len(train_loader)
    >>> clr = cyclical_lr(step_size, min_lr=0.001, max_lr=0.005)
    >>> scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
    >>> # some other operations
    >>> scheduler.step()
    >>> optimizer.step()
    """

    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2. ** (x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma ** (x)
            scale_mode = 'iterations'
        else:
            raise ValueError('The {0} is not valid value!'.format(mode))
    else:
        scale_fn = scale_func
        scale_mode = scale_md

    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError('The {} is not valid value!'.format(scale_mode))

    return lr_lambda


def test(testloader, model):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    for data in testloader:
        images, labels = data
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, labels.cuda()).item())
        del images, labels, outputs

    return test_loss / len(testloader.dataset), correct / total


def test_train(trainloader, model):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    for data in trainloader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, Variable(labels.cuda())).item())
        del images, labels, outputs

    return test_loss / len(trainloader.dataset), correct / total


def test_train_sample(trainloader, model, n_minibatches=1):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    length = 0

    trainiter = iter(trainloader)
    indexes = [0]

    while len(indexes) - 1 != n_minibatches:
        num = rm.randint(0, len(trainiter) - 2)  # Skip last minibatch
        if num not in indexes:
            indexes.append(num)

    indexes = sorted(indexes)

    for i in range(1, n_minibatches + 1):
        for j in range(indexes[i] - indexes[i - 1] - 1):
            trainiter.next()

        images, labels = trainiter.next()
        length += len(images)

        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, Variable(labels.cuda())).item())
        del images, labels, outputs

    return test_loss / total, correct / total


def restart(model, optimizer):
    print('prova')
        # if args.patience > 0 and epoch_counter >= args.patience and val_loss[epoch] - val_loss[epoch - args.patience] > 0:
        #     epoch_counter = 0
        #     minimum = 1
        #
        #     del model
        #     del optimizer
        #
        #     model = load_net(net=args.net, dataset_name=args.dataset)
        #     if args.gpu:
        #         model = model.cuda()
        #
        #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False)


# Method to train the network following the CLR policy. See also "def cyclical_lr(...)"
def train_clr(trainloader, model, optimizer, criterion, scheduler, plot):
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        for param_group in optimizer.param_groups:
            plot.append(param_group['lr'])  # deb

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        scheduler.step()
        optimizer.step()

        del inputs, labels, outputs


# Developping method of the minibatch persistency one, with a more fine controls of weight updates
# and a policy that changes the lr at each successive pass of the same minibatch
# lr_k = base_lr * 1/k 	(could account for overfitting)
# the standard minibatch persistency method does not have the term 1/k
def alt_train(trainloader, model, criterion, optimizer, gamma, plot, epoch):
    model.train()
    v_t = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if i == 0:
            for k, f in enumerate(model.parameters(), 0):
                # f.data.sub_(0.1 * f.grad.data)
                v_t.append(0.1 * f.grad.data.clone())
        else:
            d = torch.tensor([0], dtype=torch.float32).cuda()
            g = torch.tensor([0], dtype=torch.float32).cuda()
            for k, f in enumerate(model.parameters(), 0):
                g = torch.cat((g, f.grad.data.clone().view(-1)), dim=-1)
                v_t[k] = 0.5 * v_t[k] + (1 - 0.5) * f.grad.data.clone()
                d = torch.cat((d, v_t[k].view(-1)), dim=-1)

            epsilon = torch.clamp(loss.clone() / torch.clamp(torch.dot(g, d), max=1e15).data, min=0.001,
                                  max=0.5 - (epoch * 0.00499))
            plot.append(epsilon.data)
            for k, f in enumerate(model.parameters(), 0):
                f.data.sub_(epsilon.data * gamma * v_t[k])
                v_t[k] = epsilon.data * gamma * v_t[k]

        del inputs, labels, outputs


# Adaptive Nesterov training method
def nest_train(trainloader, model, criterion, optimizer, plot):
    model.train()
    v_t = []
    m = 0.1
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        if i == 0:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for k, f in enumerate(model.parameters(), 0):
                v_t.append(0.1 * f.grad.data.clone())
        else:
            for k, f in enumerate(model.parameters(), 0):
                f.data.sub_(0.5 * v_t[k])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            g = torch.tensor([0], dtype=torch.float32).cuda()
            for k, f in enumerate(model.parameters(), 0):
                g = torch.cat((g, f.grad.data.clone().view(-1)), dim=-1)

            epsilon = torch.clamp(loss.clone() / torch.clamp(torch.dot(g, g), max=1e10).data, min=0.001, max=m)
            plot.append(epsilon.data)

            # d = torch.distributions.normal.Normal(torch.Tensor([0.0]), torch.Tensor([torch.std(g)]))

            for k, f in enumerate(model.parameters(), 0):
                # if i%5 == 0:
                # f.data.sub_(f.grad.data.add_(d.sample().cuda()) * epsilon.data)
                # else:
                f.data.sub_(f.grad.data * epsilon.data)
                v_t[k] = 0.5 * v_t[k] + f.grad.data * epsilon.data

        del inputs, labels, outputs


# Adaptive Nesterov training method, with passing past update vector v_t from one epoch to the following one
# practical results are almost identical to "def nest_train(...)"
def pass_nest_train(trainloader, model, criterion, optimizer, gamma, plot, epoch, v_t):
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        if i == 0 and epoch == 0:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for k, f in enumerate(model.parameters(), 0):
                v_t.append(0.1 * f.grad.data.clone())
        else:
            for k, f in enumerate(model.parameters(), 0):
                f.data.sub_(0.5 * v_t[k])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            g = torch.tensor([0], dtype=torch.float32).cuda()
            for k, f in enumerate(model.parameters(), 0):
                g = torch.cat((g, f.grad.data.clone().view(-1)), dim=-1)

            epsilon = torch.clamp(loss.clone() / torch.clamp(torch.dot(g, g), max=1e10).data, min=0.001, max=0.1)
            plot.append(epsilon.data)

            for k, f in enumerate(model.parameters(), 0):
                f.data.sub_(f.grad.data * epsilon.data)
                v_t[k] = 0.5 * v_t[k] + f.grad.data * epsilon.data

        del inputs, labels, outputs
