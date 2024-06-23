from utils_libs import *

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

def avg_mdls(clnt_list, round, path):
    global_sum = OrderedDict()
    for clnt_id in set(clnt_list):
        new_dict = torch.load(path + f"clnt_models/clnt{clnt_id}_mdl{round}.pt")
        for key, value in new_dict.items():
            if key in global_sum:
                global_sum[key] += value
            else:
                global_sum[key] = value
    n_clnt = len(set(clnt_list))
    global_mdl = avg_dicts(global_sum, n_clnt)
    return global_mdl

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
    

PATH = "C:/Users/Sungmin Kang/Desktop/FL_implementation/Distributed_FL/"
tst_loss_list = np.load(PATH + 'results/test_loss.npy')
tst_acc_list = np.load(PATH + 'results/test_acc.npy')
    
plot_graph(data_name='Test Loss', data=tst_loss_list, cnt=9)
plot_graph(data_name='Test Accuracy', data=tst_acc_list, cnt=9)

