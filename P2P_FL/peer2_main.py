# Peer2: Client role in TCP connection
# Peer2: Having Even number dataset

from utils_libs import *
from local_train import *
from p2p_comm import *

HOST = ''
PORT = 8888
BUFFER_SIZE = 4096
SEPARATOR = "--SEP--"
PATH = "./FL_implementation/P2P_FL/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
peer_id = '2'

# Hyperparameters for local training
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


###  Main Part  ###

# Receiving Peer1 Model, current round
msg = input("\n\nReceive model from peer1 : ")
curr_round = peer2_recv_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, path=PATH)

# Local Training
local_model, loss_list, acc_list = local_train(path=PATH, peer_id=peer_id, device=device,
                                            batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=curr_round)
torch.save(local_model.state_dict(), PATH + f'peer2/local_models/peer2_mdl{curr_round}_2.pt')

# Sending Local Model to Peer1
msg = input('Send Local trained model to peer1 : ')
peer2_send_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, round=curr_round, path=PATH)

# Averaging Peer1 & Peer2 models
new_mdl = avg_mdls(peer_id=peer_id, round=curr_round, path=PATH)
torch.save(new_mdl, PATH + f"peer2/local_models/peer2_mdl{curr_round+1}_1.pt")

# Evaluating
new_model = Net()
new_model.load_state_dict(torch.load(PATH + f"peer2/local_models/peer2_mdl{curr_round+1}_1.pt"))
loss, acc = evaluate(device=device, model=new_model, criterion=criterion, test_loader=tst_loader)
print(f'>>> Round {curr_round} - Test Loss: {loss:.4f}  Test Accuracy: {acc:.2f}%')

# End of round
print(f"----------------------- Round {curr_round} End ------------------------\n")