import glob
import numpy as np

networks, folders, methods = ['alexnet', 'vgg16', 'resnet34'], [], []
epochs = 100
results = {}

for net in networks:
    folder = f'../results/svrnvidia/sgd_sa/cifar10/nodataum/{net}/not_crazy'
    results[net] = {}
    for filename in glob.glob(f'{folder}/*.npz'):
        filename_filtered = '.'.join(filename.split("/")[-1].split(".")[0:-1])
        methods.append(filename_filtered)
        results[net][filename_filtered] = np.load(filename)

methods = set(methods)

with open('../results/svrnvidia/sgd_sa/cifar10/nodataum/set.csv', 'w') as out:
    out.write('method')
    for net in networks:
        net = ''.join(filter(lambda x: not x.isdigit(), net))
        out.write(f',{net}loss,{net}acc')
    out.write('\n')
    for method in methods:
        out.write(f'{method}')
        for net in networks:
            out.write(f",{round(np.min(results[net][method]['validation_loss']), 6)},"
                      f"{round(np.max(results[net][method]['validation_accuracy'])*100, 2)}")
        out.write('\n')
