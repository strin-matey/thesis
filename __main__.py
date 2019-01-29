import time
import numpy as np
from src.utils import *

args = parse_arguments()

#set_seed(args.seed)

# Load train and test data sets
train_loader, test_loader = load_dataset(dataset_name=args.dataset, minibatch=args.batch_size,
                                         num_workers=args.numworkers, cluster=args.cluster)

model = load_net(net=args.net, dataset_name=args.dataset)
if args.gpu:
    model = model.cuda()

val_loss, val_acc, train_loss, train_acc, times = [['Nothing' for i in range(args.epochs)] for j in range(5)]
probabilities, na, ac, drop_eps, bm = {}, [], [], {}, []
start = time.clock()

epsilon = args.epsilon

if args.ssa:
    for name, param in model.named_parameters():
        print(name, param.size())
    print(count_parameters(model))

    temperature = args.temperature
    cooling_factor = args.cooling_factor
    accepted_ratio, accepted, not_accepted = None, None, None

    print("Bound:", args.accepted_bound)
    for epoch in range(args.epochs):
        print("Epoch: ", epoch, ", epsilon: ", epsilon, ", temperature:", temperature)

        not_accepted, accepted, probs = stochastic_simulated_annealing(train_loader, model, epsilon, temperature)

        probabilities[epoch] = probs
        na.append(not_accepted)
        ac.append(accepted)
        bm.append((len(train_loader)-(accepted+not_accepted)) / len(train_loader))
        times[epoch] = time.clock() - start

        # Training information
        print(f"Better moves: {len(train_loader)-(accepted+not_accepted)}/{len(train_loader)},"
              f" Accepted: {accepted}, Not accepted: {not_accepted}")
        # Evaluation on training set
        train_loss[epoch], train_acc[epoch] = test_train_sample(train_loader, model)
        print("Training loss: ", train_loss[epoch])
        print("Training accuracy:", train_acc[epoch])

        # Evaluation on validation set
        val_loss[epoch], val_acc[epoch] = test(test_loader, model)
        print("Validation loss: ", val_loss[epoch])
        print("Validation accuracy: ", val_acc[epoch])

        temperature *= cooling_factor
        temperature = max(1e-10, temperature)
        accepted_ratio = 1 - (not_accepted + accepted) / len(train_loader)
        print(f"Accepted ratio:{accepted_ratio}")
        if accepted_ratio < args.accepted_bound:
            drop_eps[epoch] = True
            epsilon /= 10
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(model, args)

    for epoch in range(args.epochs):
        print("Epoch: ", epoch)

        train(train_loader, model, optimizer, criterion, gpu=args.gpu)

        val_loss[epoch], val_acc[epoch] = test(test_loader, model)
        train_loss[epoch], train_acc[epoch] = test_train_sample(train_loader, model)

        print("Training loss: ", train_loss[epoch])
        print("Training accuracy:", train_acc[epoch])
        print("Validation loss: ", val_loss[epoch])
        print("Validation accuracy: ", val_acc[epoch])

        times[epoch] = time.clock() - start

#save_model(model, epoch, 'model.pt')

np.savez(f'results_dataaum_{args.optimizer}_{args.dataset}_{args.net}_{args.batch_size}_{args.epochs}_lr{args.lr}_bound{args.accepted_bound}_eps{args.epsilon}',
         times=times,
         validation_accuracy=val_acc,
         validation_loss=val_loss,
         train_accuracy=train_acc,
         train_loss=train_loss,
         epochs=np.array([args.epochs]),  # TODO inserire job-ID
         lr=np.array([args.lr]),
         momentum=np.array([args.momentum]),
         probabilities=probabilities,
         na=na,
         ac=ac,
         better_moves=bm,
         drop_eps=drop_eps,
         epsilon=args.epsilon)

del model
