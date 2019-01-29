import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse
import random as rm
import math

from src.resnet import *
from src.alexnet import *
from src.vgg import *
from src.BasicNet import *
from src.ConvNet import *


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
    parser.add_argument('--epsilon', type=float, default=10e-4, metavar='E',
                        help='SGA epsilon')
    parser.add_argument('--accepted_bound', type=float, default=0.5, metavar='R',
                        help='Bound accepted ratio')
    parser.add_argument('--patience', type=int, default=0, metavar='P',
                        help='Early-stopping restart')
    parser.add_argument('--temperature', type=float, default=1, metavar='T',
                        help='Temperature')
    parser.add_argument('--cooling_factor', type=float, default=0.97, metavar='C',
                        help='Cooling factor')
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
    parser.add_argument('--ssa', action='store_true', default=False)

    return parser.parse_args()


def set_seed(seed):
    # Reproducible runs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def choose_optimizer(model, args):

    """

    :param model: The model to be trained
    :param args: Command line arguments
    :return: The optimizer

    """

    opt = args.optimizer

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def train(train_loader, model, optimizer, criterion, gpu=True):

    """

    :param train_loader: The training set loader
    :param model: The model to be optimized
    :param optimizer:
    :param criterion:
    :param gpu:
    :return:

    """

    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        del inputs, labels, outputs


def load_dataset(dataset_name="cifar10", minibatch=512, num_workers=2, cluster=False, drop_last=False):

    """
    Load the dataset
    :param dataset_name: Available options cifar10, cifar100, MNIST
    :param minibatch: Minibatch size
    :param num_workers: Number of workers to load the dataset
    :param cluster:
    :return: Train loader and test loader
    """

    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4), 	#uncomment for data augmentation
            #transforms.RandomHorizontalFlip(),      #uncomment for data augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        train_set = torchvision.datasets.CIFAR10(root='/home/met/PycharmProjects/thesis/data/cifar10' if not cluster else '/home/stringherm/thesis/data/cifar10', train=True,
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch,
                                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
        test_set = torchvision.datasets.CIFAR10(root='/home/met/PycharmProjects/thesis/data/cifar10' if not cluster else '/home/stringherm/thesis/data/cifar10', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=minibatch,
                                                 shuffle=True, num_workers=num_workers, drop_last=drop_last)
        return train_loader, test_loader
    elif dataset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))])
        trainset = torchvision.datasets.CIFAR100(root='/home/met/PycharmProjects/thesis/data/cifar100' if not cluster else '/home/stringherm/thesis/data',
                                                 train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch,
                                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
        testset = torchvision.datasets.CIFAR100(root='/home/met/PycharmProjects/thesis/data/cifar100' if not cluster else '/home/stringherm/thesis/data',
                                                train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch,
                                                 shuffle=False, num_workers=num_workers, drop_last=drop_last)
        return trainloader, testloader
    elif dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='/home/met/PycharmProjects/thesis/data/mnist' if not cluster else '/home/stringherm/thesis/data/minst', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch,
                                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
        testset = torchvision.datasets.MNIST(root='/home/met/PycharmProjects/thesis/data/mnist' if not cluster else '/home/stringherm/thesis/data/mnist', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch,
                                                 shuffle=False, num_workers=num_workers, drop_last=drop_last)
        return trainloader, testloader
    elif dataset_name == "fashion-mnist":
        # Loading dataset
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.FashionMNIST(root='/home/met/PycharmProjects/thesis/data/fashion-mnist' if not cluster else '/home/stringherm/thesis/data/fashion-minst',
                                              train=True, download=True, transform=transform)

        test_dataset = torchvision.datasets.FashionMNIST(root='/home/met/PycharmProjects/thesis/data/fashion-mnist' if not cluster else '/home/stringherm/thesis/data/fashion-minst',
                                                        train=False, download=True, transform=transform)
        # Loading dataset into dataloader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=minibatch,
                                                   shuffle=True, num_workers=num_workers, drop_last=drop_last)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=minibatch,
                                                  shuffle=False, num_workers=num_workers, drop_last=drop_last)
        return train_loader, test_loader


def load_net(net="alexnet", dataset_name="cifar10"):

    """

    :param net:  Available options AlexNet, BasicNet, ConvNet, ResNet 18, ResNet34, ResNet 50, ResNet 101
                VGG11, VGG16, VGG19
    :param dataset_name:
    :return:

    """
    num_classes = 10
    if dataset_name == "cifar10" or dataset_name == "mnist" or dataset_name == "fashion-mnist":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    elif dataset_name == "imagenet":
        num_classes = 1000

    if net == "alexnet":
        return alexnet(num_classes)
    elif net == "basicnet":
        return BasicNet()
    elif net == "convnet":
        return ConvNet()
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


def test(test_loader, model):

    """
    Evaulate over tuhe test set
    :param test_loader:
    :param model:
    :return:

    """

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    for data in test_loader:
        images, labels = data
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, labels.cuda()).item())
        del images, labels, outputs

    return test_loss / len(test_loader.dataset), correct / total


def test_train(train_loader, model):

    """
    Evaluate over training set
    :param train_loader:
    :param model:
    :return:
    """

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    for data in train_loader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, Variable(labels.cuda())).item())
        del images, labels, outputs

    return test_loss / len(train_loader.dataset), correct / total


def test_minibatch(images, labels, model):

    """
    Evaluate over a single minibatch.
    This functions is used in S
    :param images:
    :param labels:
    :param model:
    :return:
    """
    model.eval()
    correct = 0
    total = labels.size(0)
    test_loss = 0

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
    test_loss += float(F.cross_entropy(outputs, labels).item())

    return test_loss / total, correct / total


def test_train_sample(train_loader, model, n_minibatches=10):

    """
    Test on a sample of
    :param train_loader:
    :param model:
    :param n_minibatches:
    :return:
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    length = 0

    train_iter = iter(train_loader)
    indexes = [0]

    while len(indexes) - 1 != n_minibatches:
        num = rm.randint(0, len(train_iter) - 2)  # Skip last minibatch
        if num not in indexes:
            indexes.append(num)

    indexes = sorted(indexes)

    for i in range(1, n_minibatches + 1):
        for j in range(indexes[i] - indexes[i - 1] - 1):
            train_iter.next()

        images, labels = train_iter.next()
        length += len(images)

        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, Variable(labels.cuda())).item())
        del images, labels, outputs

    return test_loss / total, correct / total


def save_model(model, epoch, path):

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, path)


def restart(trainloader, model, args, optimizer, epsilon=0.0002):
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

    train_cost, train_accuracy = test_train_sample(trainloader, model, n_minibatches=10)

    if train_cost < epsilon:
        print("Restart")

        del model
        del optimizer

        model = load_net(net=args.net, dataset_name=args.dataset)
        if args.gpu:
            model = model.cuda()

        optimizer = choose_optimizer(model, args)

    return model, optimizer, train_cost, train_accuracy


def stochastic_simulated_annealing(train_loader, model, epsilon, T, gpu=True):
    model.train()

    not_accepted, accepted = 0, 0
    probabilities = []

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        initial_loss = test_minibatch(inputs, labels, model)[0]
        print(f"Initial loss: {initial_loss}")
        # List used to keep the move to get back to the initial point
        inverse = []

        # First move
        for param in model.parameters():
            # Replicate the tensor
            tensor_size = param.data.size()
            move = torch.zeros(tensor_size)
            # Send it to the GPU
            if gpu:
                move = move.cuda()
            # Generate move
            move = move.normal_(std=epsilon)
            # Step back is saved
            inverse.append(move.mul(-1))
            # Move the parameters
            param.data.add_(move)
        # Evaluate the loss
        first_loss = test_minibatch(inputs, labels, model)[0]
        print(f"First loss: {first_loss}")
        # Second move
        for k, param in enumerate(model.parameters()):
            param.data.add_(inverse[k].mul(2))
            inverse[k].mul_(-1)
        second_loss = test_minibatch(inputs, labels, model)[0]
        print(f"Second loss: {second_loss}")
        # Get back if the first move is better
        if first_loss < second_loss:
            for k, param in enumerate(model.parameters()):
                param.data.add_(inverse[k].mul(2))
                inverse[k].mul_(-1)
            new_loss = first_loss
        else:
            new_loss = second_loss

        if new_loss > initial_loss:
            probabilities.append(math.exp(- (new_loss - initial_loss) / T))

        # Reject worse solution according to the standard formula
        if new_loss > initial_loss and math.exp(- (new_loss - initial_loss) / T) < rm.random():
            not_accepted += 1
            for k, param in enumerate(model.parameters()):
                param.data.add_(inverse[k])
            new_loss = initial_loss
        elif new_loss > initial_loss:
            accepted += 1
        print(f"New loss: {new_loss}")
        print(f"Real final loss: {test_minibatch(inputs, labels, model)[0]}")

        del move, inverse, inputs, labels

    return not_accepted, accepted, probabilities


def simulated_annealing(trainloader, model, initial_accuracy, epsilon, temperature, gpu=True):
    model.train()

    # List used to keep the move to get back to the initial point
    inverse = []

    # First move
    for param in model.parameters():
        # Replicate the tensor
        tensor_size = param.data.size()
        move = torch.zeros(tensor_size)
        # Send it to the GPU
        if gpu:
            move = move.cuda()
        # Generate move
        move = move.uniform_(-1, 1).mul(epsilon) # * param.data
        # Stepback is saved
        inverse.append(move.mul(-1))
        # Move the parameters
        param.data.add_(move)
    # Evaluate the accuracy
    first_accuracy = test_train(trainloader, model)[1]
    # print("First move accuracy: ", first_accuracy)

    # Second move
    for k, param in enumerate(model.parameters()):
        param.data.add_(inverse[k].mul(2))
        inverse[k] = inverse[k].mul(-1)
        print(param.size())
    second_accuracy = test_train(trainloader, model)[1]
    # print("Second move accuracy:", second_accuracy)

    # Get back if the first accuracy is better
    if first_accuracy > second_accuracy:
        for k, param in enumerate(model.parameters()):
            param.data.add_(inverse[k].mul(2))
            inverse[k] = inverse[k].mul(-1)
        new_accuracy = first_accuracy
    else:
        new_accuracy = second_accuracy

    # Accept a worse solution according to temperature
    if new_accuracy < initial_accuracy and rm.uniform(0, 1) > temperature:
        for k, param in enumerate(model.parameters()):
            param.data.add_(inverse[k])
        new_accuracy = initial_accuracy

    del move, inverse
    # print("Final accuracy:", new_accuracy)
    return new_accuracy

