# Main code for client

from utils_libs import *
from client_train import *
from client_comm import *

# Dataset already given to each clients after splitting the dataset

# Change Variables
PATH = './FL_implementation/Distributed_FL/'
clnt_id = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice status: {device}")

# Variables to Connect with Server
HOST = ''
PORT = 8888
SEPARATOR = "--SEP--"
BUFFER_SIZE = 4096

# Hyperparameters for local training
batch_size = 16
learning_rate = 1e-6
epochs = 5

# Connect with Server and receive current round from server
msg = input('\nReceive Global model from server? type "Y" or "yes" or "y": ')
if msg == 'Y' or 'y' or 'yes':
    print('\nConnecting with Server ...')
    curr_round = recv_global_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR,
                    clnt_id=clnt_id, path=PATH)

    # Local Training
    time.sleep(1)
    local_model, loss_list, acc_list = local_train(path=PATH, clnt_id=clnt_id, device=device,
                                                batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=curr_round)

    # Saving Local Model
    torch.save(local_model.state_dict(), PATH+f'clnt_models/clnt{clnt_id}_mdl{curr_round}.pt')

    # Plotting Local Training Loss and Accuracy curves
    # plot_graph(data_name = 'Loss', data = loss_list, epochs = epochs)
    # plot_graph(data_name = 'Accuracy', data = acc_list, epochs = epochs)

    # Sending Local model to Server
    msg = input('Send the local model to the server? type "Y" or "yes" or "y": ')
    send_local_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR,
                clnt_id=clnt_id, round=curr_round, path=PATH, msg=msg)

else:
    print('Join in Next Round\n')