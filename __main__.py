from src.utils import *

args = parse_arguments()

set_seed(args.gpu_number)

set_device(args.gpu_number)
print(f"Gpu number: {args.gpu_number}")

# Load train and test data sets
train_loader, test_loader = load_dataset(dataset_name=args.dataset, minibatch=args.batch_size,
                                         num_workers=args.numworkers, cluster=args.cluster,
                                         dataaum=args.dataaum)

model = load_net(net=args.net, dataset_name=args.dataset)
if args.gpu:
    model = model.cuda()

if args.train_mode == 'ssa':
    train_ssa(train_loader, test_loader, model, args)
elif args.train_mode == 'sgd_sa':
    train_SGD_SA(train_loader, test_loader, model, args)
elif args.train_mode == 'sgd' or args.train_mode == 'scheduled_sgd':
    train_sgd(train_loader, test_loader, model, args)
