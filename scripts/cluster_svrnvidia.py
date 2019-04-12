from fabric import Connection
import getpass
import time
import datetime

'''
    Questo script può essere utilizzato per semplificare l'utilizzo del cluster.
    Intende emulare il comportamento di SGE, utilizzato nel cluster del DEI.
    
    Permette le seguenti cose:
        - Caricare i file modificati localmente in modo automatico (vd. lista "files")
        - Creare una cartella di output unica per ogni job
        - Salvare tutti i comandi nel file commands.job che lo script inserisce
          nella cartella appena creata
        - Lanciare il job in una sessione di screen in modo tale da potersi scollegare in modo sicuro
        - Tutti i file di output (stderr, stdin) vengono salvati in una serie di file nella cartella di output creata
    
    Come utilizzarlo:
        - Al primo utilizzo è necessario mettere il proprio progetto in una cartella remota.
            Esempio da terminale remoto:
            mkdir project
            cd project
            git clone ... 
        - Inserire nello script:
            - Il proprio nome utente
            - La cartella di root del progetto in locale
            - La cartella di root del progetto remoto
        - Il file eseguibile per essere invocato correttamente
          deve essere contenuto nella cartella root e deve essere chiamato: __main__.py
        - Inserire nella lista "files" tutti i file che vengono aggiornati prima di lanciare il job
'''

# User variables
username = 'stringherm'
local_project_root = '/home/met/PycharmProjects/thesis'  # Without ending slash
remote_project_root = f'/home/{username}/thesis'  # Without ending slash
files = ['__main__.py', 'src/utils.py', 'src/vgg.py', 'src/resnet.py',
         'src/alexnet.py', 'src/BasicNet.py', 'src/ConvNet.py', 'src/squeezenet.py',
         'src/mobilenet.py']

pswd = getpass.getpass('Password: ')

# Create a custom folder
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')  # Custom
identifier = abs(hash(timestamp)) % (10 ** 8)  # This identifier will be used for the screen name too
output_folder = f'run__{timestamp}_{identifier}'

# Create the commands file
# All the options to loop (Cartesian product)
datasets = ['fashion-mnist', 'mnist']
nets = ['vgg16']
batch_sizes = [256, 512, 1024]
epsilon = ['-1']
bounds = [0.7]
lrs = [0.1]
optimizer = 'momentum'
epochs = 1000
train_modes = ['ssa']

commands = "source /nfsd/opt/anaconda3/anaconda3.sh\n" \
           "conda activate /nfsd/opt/anaconda3/tensorflow\n"

# For each command a different output file will be created in the remote output folder
# This part must be modified according to your needs
cnt = 0
for d in datasets:
    for net in nets:
        for batch_size in batch_sizes:
            for eps in epsilon:
                for lr in lrs:
                    for train_mode in train_modes:
                        # Formatting/constructing the instruction to be given:
                        commands += f"time python3 -u {remote_project_root}/__main__.py " \
                                    f" --cluster --gpu --gpu_number 1"

                        # Options to be added:
                        commands += f" --dataset {d}"
                        commands += f" --outfolder {output_folder}"
                        commands += f" --epochs {epochs}"
                        commands += f" --batch-size {batch_size}"
                        commands += f" --net {net}"
                        commands += f" --lr {lr} --momentum 0 "
                        commands += f" --restart_epsilon {eps} "
                        commands += f" --optimizer {optimizer}"
                        commands += f" --train_mode {train_mode}"
                        commands += f" --cooling_factor 0.97"

                        commands += f" &>> {identifier}_{cnt}.txt\n"
                        cnt += 1

# Create a connect to the svrnvidia machine and send the job
with Connection(f'{username}@login.dei.unipd.it', connect_kwargs={"password": pswd}) as proxy:
    with Connection(f'{username}@svrnvidia', gateway=proxy, connect_kwargs={"password": pswd}) as c:

        print("Uploading files: ")
        # Update the files on the cluster
        for file in files:
            print(f'{local_project_root}/{file} >>> {remote_project_root}/{file}')
            c.put(f'{local_project_root}/{file}', f'{remote_project_root}/{file}')

        # Create out and custom folder
        print(f"Creating custom folder {output_folder}")
        c.run(f'mkdir {remote_project_root}/out -p')
        c.run(f'mkdir {remote_project_root}/out/{output_folder}')

        # Commands.job file is saved
        print("Saving commands.sh file")
        c.run(f'echo "{commands}" > {remote_project_root}/out/{output_folder}/commands.sh')

        c.run(f'cd {remote_project_root}/out/{output_folder} \n'
              f'chmod +x commands.sh \n'
              f'screen -dmSL {identifier} bash -c ./commands.sh')

        print(f'Command launched with screen id: {identifier}')


