# FL_Socket: Federated Learning Implementation using Socket Communication

### Requirements

Please install the required packages via:

```pip install -r requirements.txt```


### Instructions

This code uses the MNIST dataset on .csv file, and './FL_implementation/data_split.py' code is used to split the MNIST dataset in i.i.d distribution adjusting the number of clients, or in two separate datasets: having only odd and even number labels. 
- MNIST training dataset is missing because of big data file. You can download csv MNIST file from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download and save as './FL_implementation/Distributed_FL/dataset/mnist/mnist_train.csv'

There are two types of Federated Learning scenarios in this code:
 
(1) Distributed_FL
- Distributed_FL scenario is a conventional horizontal FL scenario with multiple clients communicating with a server. 
- The initial global model is saved as './FL_implementation/Distributed_FL/global_models/global_mdl1.pt'. This file is necessary for running the code. 
- The clients uses 'client_comm.py', 'client_train.py', 'client_main.py', 'utils_libs.py' code files and directories in './FL_implementation/Distributed_FL.
- The server uses 'server_comm.py', 'server_aggregate.py', 'server_main.py', 'utils_libs.py' code files and directories in './FL_implementation/Distributed_FL.
- Handling multiple clients is done through select(), and threading.

(2) P2P_FL
- P2P_FL scenario is a FL scenario with peer-to-peer structure, without having a server. This code is implemented with 2-peer P2P structure. 
- There is no initial model to be saved for P2P_FL. The initial model for each peer is created after training on local datasets.
- Peer 1 acts as a server, and peer 2 acts as a client in socket connection, but the role in Federated Learning is the same.
- Peer 1 uses 'local_train.py', 'p2p_comm.py', 'peer1_main.py', './dataset', './peer1', './results' in './FL_implementation/P2P_FL.
- Peer 2 uses 'local_train.py', 'p2p_comm.py', 'peer2_main.py', './dataset', './peer2', './results' in './FL_implementation/P2P_FL.
