# Peer1: Server role in TCP connection
# Peer1: Having Odd number dataset

from utils_libs import *
from local_train import *
from p2p_comm import *

HOST = ''
PORT = 8888
BUFFER_SIZE = 4096
SEPARATOR = "--SEP--"
PATH = "./FL_implementation/P2P_FL/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
peer_id = '1'

# Hyperparameters for training
rounds = 5
batch_size = 16
learning_rate = 1e-6
epochs = 5

# Test dataset, evaluation configuration
tst_df = pd.read_csv(PATH + 'dataset/mnist/mnist_test.csv')
tst_pixels = tst_df.iloc[:, 1:].to_numpy()
tst_labels = tst_df.iloc[:, 0].to_numpy()

tst_set = mnist_dataset(tst_pixels, tst_labels,
                        transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
tst_loader = DataLoader(tst_set, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
tst_loss_list = []
tst_acc_list = []


# P2P Federated Learning Process
for round in range(1, rounds+1):
    print(f"\n---------------------- Round {round} Start ----------------------")
    
    # Local Training
    local_model, loss_list, acc_list = local_train(path=PATH, peer_id=peer_id, device=device,
                                               batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=round)
    torch.save(local_model.state_dict(), PATH + f'peer1/local_models/peer1_mdl{round}_2.pt')
    
    # Sending Local Model to Peer2
    msg = input('Send Local trained model to peer2 : ')
    peer1_send_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, round=round, path=PATH)
    
    
    # Receiving Peer2 Model
    msg = input("Receive model from peer2 : ")
    connected = peer1_recv_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, path=PATH)
    
    
    # Averaging Peer1 & Peer2 models
    if connected:
        new_mdl = avg_mdls(peer_id=peer_id, round=round, path=PATH)
        torch.save(new_mdl, PATH + f"peer1/local_models/peer1_mdl{round+1}_1.pt")
    
    # Check new model and copy if None
    check_new_mdl(peer_id=peer_id, round=round+1, path=PATH)
    
    # Evaluating
    new_model = Net()
    new_model.load_state_dict(torch.load(PATH + f"peer1/local_models/peer1_mdl{round+1}_1.pt"))
    loss, acc = evaluate(device=device, model=new_model, criterion=criterion, test_loader=tst_loader)
    print(f'>>> Round {round} - Test Loss: {loss:.4f}  Test Accuracy: {acc:.2f}%')
    
    # Plotting Loss and Accuracy curve
    tst_loss_list.append(loss)
    tst_acc_list.append(acc)
    
    # Saving Test Loss and Accuracy Data
    np.save(PATH+'results/test_loss.npy', tst_loss_list)
    np.save(PATH+'results/test_acc.npy', tst_acc_list)
    
    # End of round
    print(f"----------------------- Round {round} End ------------------------")
    
# Plotting, Saving Test Loss and Accuracy
plot_graph(data_name='Test Loss', data=tst_loss_list, cnt=rounds)
plot_graph(data_name='Test Accuracy', data=tst_acc_list, cnt=rounds)
    
print('\n----------------- End of All Learning Process -----------------\n')