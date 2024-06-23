from utils_libs import *
from server_aggregate import *
from server_comm import *

HOST = ''
PORT = 8888
BUFFER_SIZE = 4096
SEPARATOR = "--SEP--"
PATH = "./FL_implementation/Distributed_FL/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rounds = 10
batch_size = 16
connection_time_limit = 18
num_max_clients = 6

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

# Federated Learning Process
for round in range(1, rounds+1):
    print(f"\n---------------------- Round {round} Start ----------------------")
    
    # Sending global model to clients
    connected_clients = send_global_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR,
                                        round=round, time_limit=connection_time_limit, path=PATH, max_clients=num_max_clients)
    
    # Receiving Local client models from clients
    msg = input("Do you want to receive local models from clients? : ")
    clnt_mdl_list = recv_local_mdls(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR,
                    time_limit=connection_time_limit, path=PATH, max_clients=num_max_clients)
    
    # Averaging client models
    n_clnts = len(set(clnt_mdl_list))
    if n_clnts != 0:
        global_mdl = avg_mdls(clnt_list=clnt_mdl_list, round=round, path=PATH)
        torch.save(global_mdl, PATH + f"global_models/global_mdl{round+1}.pt")
    
    # Evaluating with global model
    check_global_mdl(round=round+1, path=PATH)
    global_model = Net()
    global_model.load_state_dict(torch.load(PATH + f"global_models/global_mdl{round+1}.pt"))
    loss, acc = evaluate(device=device, model=global_model, criterion=criterion, test_loader=tst_loader)
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