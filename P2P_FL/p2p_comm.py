from utils_libs import *

def check_new_mdl(peer_id, round, path):
    filename = path + f"peer{peer_id}/local_models/peer{peer_id}_mdl{round}_1.pt"
    
    # No model from previous round
    if os.path.isfile(filename) == False:   
        shutil.copy((path + f"peer{peer_id}/local_models/peer{peer_id}_mdl{round-1}_2.pt"), filename)
 
             
### Sending & Receiving functions for peer1 (TCP server) ###

# Sending current round, peer1 model
def peer1_send_mdl(host, port, bufsize, sep, round, path):
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Establish Communication with another peer and send model
    serverSocket.bind((host, port))
    serverSocket.listen()
    sock, addr = serverSocket.accept()
    filename = path + "peer1/local_models/peer1_mdl{}_2.pt".format(round)          
    filesize = os.path.getsize(filename)
    sock.sendall(f"{round}{sep}{filename}{sep}{filesize}".encode())
    time.sleep(1)   # time buffer for smooth file communication
    with open(filename, "rb") as f:
        while True:
            bytes_read = f.read(bufsize)
            if not bytes_read: break
            sock.sendall(bytes_read)
    print('Successfully sent local model to peer2\n')
    sock.close()
    serverSocket.close()

# Receiving model from peer2
def peer1_recv_mdl(host, port, bufsize, sep, path):
    connected = False
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind((host, port))
    serverSocket.listen()   
    sock, addr = serverSocket.accept()
    received = sock.recv(bufsize).decode()
    if sep in received:
        connected = True
        filename, filesize = received.split(sep)
        filename = os.path.basename(filename)
        filesize = int(filesize)
        with open(path+"peer1/recvd_models/"+filename, "wb") as f:
            while True:
                bytes_read = sock.recv(bufsize)
                if not bytes_read: break
                f.write(bytes_read)
        print('Successfully received model from peer2')
    # End Communication
    sock.close()
    serverSocket.close()
    return connected
    
    
### Sending & Receiving functions for peer2 (TCP client) ###
def peer2_send_mdl(host, port, bufsize, sep, round, path):
    # Creating Socket
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.settimeout(10)

    # Connecting with Server Socket
    clientSocket.connect((host, port))
    
    # send local model
    filename = path + f"peer2/local_models/peer2_mdl{round}_2.pt"
    filesize = os.path.getsize(filename)
    clientSocket.sendall(f"{filename}{sep}{filesize}".encode())
    time.sleep(1)
    with open(filename, "rb") as f:
        while True:
            bytes_read = f.read(bufsize)
            if not bytes_read: break
            clientSocket.sendall(bytes_read)
    clientSocket.close()
    print('Successfully sent local model to peer1\n\n')


# Receiving current round and peer1 model
def peer2_recv_mdl(host, port, bufsize, sep, path):
    # Creating Socket
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.settimeout(10)

    # Connecting with Server Socket
    try:
        clientSocket.connect((host, port))
    except ConnectionRefusedError:
        print('Cannot connect to Server')
        print('Check the host and port number of the server')
        print('Check your client id and your file path')

    # Receiving current round and peer1 model
    try:
        received = clientSocket.recv(bufsize).decode()
        curr_round, filename, filesize = received.split(sep)
        print('\n--------------------- Current round is:', curr_round, '---------------------')
        filename = os.path.basename(filename)
        filesize = int(filesize)
        with open(path+"peer2/recvd_models/"+filename, "wb") as f:
            while True:
                bytes_read = clientSocket.recv(bufsize)
                if not bytes_read: break
                f.write(bytes_read)
        if filesize:
            print("Successfully received model from peer1")
    except ConnectionRefusedError:
        print('Connection Lost. Participate in the next round.')
    except ValueError:
        print('Invalid value due to distracted communication')
    
    clientSocket.close()
    return int(curr_round)