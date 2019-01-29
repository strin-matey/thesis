# Plot libraries and tables
import matplotlib.pyplot as plt
import glob
import numpy as np

results = {}
for filename in glob.glob('../results/SSA/fashion-mnist/vgg16/epsilon_costante/*.npz'):
    filename_filtered = ''.join(filename.split("/")[-1].split(".")[0:-1])
    print(filename_filtered)
    results.update({filename_filtered: np.load(filename)})

fig, ax = plt.subplots()

for name, result in results.items():
    drop_eps = result['drop_eps'][()]

    n_epochs = len(result['train_loss'])
    plt.ylim([0, np.max(result['train_loss'])])
    plt.title('Loss comparison')
    plt.xlabel('Epochs')
    ax.plot(np.arange(0, n_epochs), result['train_loss'], '--', label=name+' train')
    ax.plot(np.arange(0, n_epochs), result['validation_loss'], '--', label=name+' validation')

    for drop in drop_eps.keys():
        plt.axvline(x=drop - 1, color='red')

ax.legend()
plt.show()

fig, ax = plt.subplots()

for name, result in results.items():
    drop_eps = result['drop_eps'][()]

    n_epochs = len(result['train_accuracy'])
    plt.title('Accuracy comparison')
    plt.xlabel('Epochs')
    ax.plot(np.arange(0, n_epochs), result['train_accuracy'], '--', label=name+' train')
    ax.plot(np.arange(0, n_epochs), result['validation_accuracy'], '--', label=name+' validation')

    for drop in drop_eps.keys():
        plt.axvline(x=drop - 1, color='red')

ax.legend()
plt.show()

fig, ax = plt.subplots()

for name, result in results.items():
    probabilities = result['probabilities'][()]
    x, y = zip(*((k, float(x)) for k in probabilities for x in probabilities[k]))
    n_epochs = len(result['train_accuracy'])
    plt.title('Number of accepted moves')
    ax.scatter(x, y, label=name)

ax.legend()
plt.show()

fig, ax = plt.subplots()

for name, result in results.items():
    na = result['na']
    ac = result['ac']

    ax.scatter(np.arange(0, len(na)) - 0.1, na, label='Worse moves not accepted', alpha=0.2)
    ax.scatter(np.arange(0, len(ac)), ac, label='Worse moves accepted', alpha=0.2)
    ax.scatter(np.arange(0, len(na)) + 0.1, np.asarray(na) + np.asarray(ac), label='Worse moves', alpha=0.2)

    for drop in drop_eps.keys():
        plt.axvline(x=drop - 1, color='red')

ax.legend()
plt.show()
