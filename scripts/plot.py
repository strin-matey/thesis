# Plot libraries and tables
import matplotlib.pyplot as plt
import glob
import numpy as np

folder = '../results/svrnvidia/sgd_sa/cifar10/nodataum/vgg16/not_crazy'
plot_name = 'vgg16_set2'
epochs = 100
results = {}
for filename in glob.glob(f'{folder}/*.npz'):
    filename_filtered = '.'.join(filename.split("/")[-1].split(".")[0:-1])
    print(filename_filtered)
    results.update({filename_filtered: np.load(filename)})

# Training loss
fig, ax = plt.subplots()

for name, result in results.items():
    n_epochs = len(result['train_loss'][0:epochs - 1])
    ax.plot(np.arange(0, n_epochs), result['train_loss'][0:epochs - 1], '--', label=name)

plt.title('Training loss comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
ax.legend()
plt.show()
fig.savefig(f'{folder}/{plot_name}_train_loss.pdf')

# Validation loss
fig, ax = plt.subplots()
for name, result in results.items():
    n_epochs = len(result['validation_loss'][0:epochs - 1])
    ax.plot(np.arange(0, n_epochs), result['validation_loss'][0:epochs - 1], '--', label=name)

n_epochs = len(result['validation_loss'])
plt.title('Validation loss comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
ax.legend()

plt.show()
fig.savefig(f'{folder}/{plot_name}_validation_loss.pdf')

# Validation accuracy
fig, ax = plt.subplots()
for name, result in results.items():
    n_epochs = len(result['validation_accuracy'][0:epochs - 1])
    ax.plot(np.arange(0, n_epochs), result['validation_accuracy'][0:epochs - 1], '--', label=name)

plt.title('Validation accuracy comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
ax.legend()
fig.savefig(f'{folder}/{plot_name}_validation_accuracy.pdf')
plt.show()

# Train accuracy
fig, ax = plt.subplots()
for name, result in results.items():
    n_epochs = len(result['train_accuracy'][0:epochs - 1])
    ax.plot(np.arange(0, n_epochs), result['train_accuracy'][0:epochs - 1], '--', label=name)

plt.title('Train accuracy comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
ax.legend()
plt.savefig(f'{folder}/{plot_name}_train_accuracy.pdf')
plt.show()

# Loss
fig, ax = plt.subplots()
for name, result in results.items():
    n_epochs = len(result['train_loss'][0:epochs - 1])
    ax.plot(np.arange(0, n_epochs), result['train_loss'][0:epochs - 1], '--', label=name+' train')
    ax.plot(np.arange(0, n_epochs), result['validation_loss'][0:epochs - 1], '--', label=name+' validation')

plt.title('Loss comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
ax.legend()
plt.savefig(f'{folder}/{plot_name}_loss.pdf')
plt.show()

# Accuracy
fig, ax = plt.subplots()
for name, result in results.items():
    n_epochs = len(result['train_accuracy'][0:epochs - 1])
    ax.plot(np.arange(0, n_epochs), result['train_accuracy'][0:epochs - 1], '--', label=name+' train')
    ax.plot(np.arange(0, n_epochs), result['validation_accuracy'][0:epochs - 1], '--', label=name+' validation')

plt.title('Accuracy comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
ax.legend()
plt.savefig(f'{folder}/{plot_name}_accuracy.pdf')
plt.show()


# SSA PLOTS

fig, ax = plt.subplots()

for name, result in results.items():
    if 'probabilities' in result:
        probabilities = result['probabilities'][()]
        x, y = zip(*((k, float(x)) for k in probabilities for x in probabilities[k]))
        n_epochs = len(result['train_accuracy'])
        ax.scatter(x, y, label=name, alpha=0.2)

plt.xlabel('Epochs')
plt.ylabel('Probability')
plt.title('Probability decay')
fig.savefig(f'{folder}/{plot_name}_probabilities.png')

plt.show()

fig, ax = plt.subplots()

for name, result in results.items():
    if 'na' in result:
        na = result['na']
        ac = result['ac']
        drop_eps = result['drop_eps'][()]

        ax.scatter(np.arange(0, len(na)) - 0.1, na, label='Worse moves not accepted', alpha=0.2)
        ax.scatter(np.arange(0, len(ac)), ac, label='Worse moves accepted', alpha=0.2)
        #ax.scatter(np.arange(0, len(na)) + 0.1, np.asarray(na) + np.asarray(ac), label='Worse moves', alpha=0.5)

        for drop in drop_eps.keys():
            plt.axvline(x=drop - 1, color='red')

ax.legend()
plt.xlabel('Epochs')
plt.ylabel('Number of moves')
plt.title('Number of accepted moves')
fig.savefig(f'{folder}/{plot_name}_moves.png')
plt.show()
