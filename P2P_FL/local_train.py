### Client code for Local Training

from utils_libs import *

# Dataset Configuration
class mnist_dataset(Dataset):
    def __init__(self, img, label, transform=None):
        self.img = img
        self.label = label
        self.transform = transform
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.img[idx,:].reshape(28, 28), dtype=torch.float).numpy()
        label = torch.tensor(int(self.label[idx]), dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, label
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_cls = 10
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.n_cls)
    
    def forward(self, x):
        x = x.view(-1, 1*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def local_train(path=None, peer_id=None, device=None, batch_size=16, learning_rate=0.001, epochs=5, curr_round=None):
    # Dataset, DataLoader configuration
    if peer_id == '1':
        df = pd.read_csv(path+f'dataset/mnist/mnist_train_odds.csv')
    else:
        df = pd.read_csv(path+f'dataset/mnist/mnist_train_even.csv')
    train_pixels = df.iloc[:, 2:].to_numpy()
    train_labels = df.iloc[:, 1].to_numpy()
    
    train_set = mnist_dataset(train_pixels,
                              train_labels,
                              transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(train_set,
                              batch_size = batch_size,
                              shuffle = True)
    
    # Initializing Model, Loss function, and Optimizer
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    # Loading averaged model parameter
    if os.path.isfile(path + f'peer{peer_id}/local_models/peer{peer_id}_mdl{curr_round}_1.pt'):
        model.load_state_dict(torch.load(path + f'peer{peer_id}/local_models/peer{peer_id}_mdl{curr_round}_1.pt'))
    
    # Training Process
    print('\n ------------- Start of Local Training -------------')
    start = time.time()
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        loss = 0
        acc = 0
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            pred = output.max(1, keepdim=True)[1]
            acc += pred.eq(label.view_as(pred)).sum().item()
        
        acc = 100.*acc/len(train_set)
        print(f"Epoch[{epoch+1}/{epochs}]\tLoss: {loss.item():.4f}\tAccuracy: {acc:.2f}%")
        loss_list.append(loss.item())
        acc_list.append(acc)
    end = time.time()
    print(' -------------- End of Local Training --------------\n')
    print(f"Training Time: {end-start:.4f} seconds\n")
    return model, loss_list, acc_list
    
    
def avg_mdls(peer_id, round, path):
    mdl_sum = OrderedDict()
    if peer_id == '1':
        peer1_dict = torch.load(path + f"peer1/local_models/peer1_mdl{round}_2.pt")
        peer2_dict = torch.load(path + f"peer1/recvd_models/peer2_mdl{round}_2.pt")
    else:
        peer1_dict = torch.load(path + f"peer2/recvd_models/peer1_mdl{round}_2.pt")
        peer2_dict = torch.load(path + f"peer2/local_models/peer2_mdl{round}_2.pt")      
    for key, value in peer1_dict.items():
        if key in mdl_sum:
            mdl_sum[key] += value
        else:
            mdl_sum[key] = value
    for key, value in peer2_dict.items():
        if key in mdl_sum:
            mdl_sum[key] += value
        else:
            mdl_sum[key] = value
    new_mdl = avg_dicts(mdl_sum, 2)
    return new_mdl

def avg_dicts(dict, n_clnt):
    for key, value in dict.items():
        dict[key] = value/n_clnt
    return dict


def evaluate(device, model, criterion, test_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss += criterion(output, label).item()
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    loss /= len(test_loader.dataset)
    tst_acc = 100.*correct/len(test_loader.dataset)
    return loss, tst_acc


def plot_graph(data_name, data, cnt):
    plt.plot(range(1, cnt+1), data)
    plt.title(f'{data_name}')
    plt.xlabel('Round')
    plt.ylabel(f'{data_name}')
    plt.show()