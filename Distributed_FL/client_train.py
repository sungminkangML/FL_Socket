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
    
def local_train(path=None, clnt_id=None, device=None, batch_size=16, learning_rate=0.001, epochs=5, curr_round=None):
    # Dataset, DataLoader configuration
    df = pd.read_csv(path+f'dataset/mnist/div_10/clnt{clnt_id}_train.csv')
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
    
    # Loading global model parameter
    if os.path.isfile(path + f'global_models/global_mdl{curr_round}.pt'):
        model.load_state_dict(torch.load(path + f'global_models/global_mdl{curr_round}.pt'))
    
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
    print(' ------------- End of Local Training -------------\n')
    print(f"Training Time: {end-start:.4f} seconds\n")
    return model, loss_list, acc_list

def plot_graph(data_name = '', data = None, epochs = 5):
    plt.plot(range(1, epochs+1), data)
    plt.title(f'Training {data_name}')
    plt.xlabel('Epoch')
    plt.ylabel(f'Training {data_name}')
    plt.show()