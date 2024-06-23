from utils_libs import *

def check_global_mdl(round, path):
    filename = path + "global_models/global_mdl{}.pt".format(round)
    
    # No global model from previous round
    if os.path.isfile(filename) == False:   
        shutil.copy((path + "global_models/global_mdl{}.pt".format(round-1)), filename)
             
                    
def send_global_mdl(host, port, bufsize, sep, round, time_limit, path, max_clients):
    connected_clients = []
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection_list = [serverSocket]
    
    # Establish Communication with Clients and distributing global models
    msg = input('Do you want to connect with clients? : ')
    print("\n >>>>>>>   Connecting with clients ...   <<<<<<<")
    try:
        serverSocket.bind((host, port))
        serverSocket.listen(max_clients)
        start = time.time()
        
        while time.time() - start < time_limit:
            read_socket, write_socket, error_socket = select.select(connection_list, [], [], time_limit)
            for sock in read_socket:
                # New Connection
                if sock == serverSocket:
                    clientSocket, addr = serverSocket.accept()
                    connection_list.append(clientSocket)
                
                # Connected Before
                else:
                    # Receiving Client ID
                    clnt_id = sock.recv(bufsize).decode()
                
                    # Client id well received
                    if clnt_id.isdigit():
                        clnt_id = int(clnt_id)
                        connected_clients.append(clnt_id)
                        sock.sendall(str(round).encode())
                        
                        # Send Global Model to connected Clients
                        try:
                            filename = path + "global_models/global_mdl{}.pt".format(round)          
                            filesize = os.path.getsize(filename)
                            sock.sendall(f"{filename}{sep}{filesize}".encode())
                            time.sleep(1)   # time buffer for smooth file communication
                            with open(filename, "rb") as f:
                                while True:
                                    bytes_read = f.read(bufsize)
                                    if not bytes_read: break
                                    sock.sendall(bytes_read)
                        except FileNotFoundError:
                            print('File Not Found. Might be the wrong path')
                        except ConnectionResetError:
                            print('Connection forcibly closed. Moving on to next round.')
                            break
                    connection_list.remove(sock)
                    sock.close()
    except ConnectionResetError:
        print('Connection forcibly closed. Moving on to next round.')
    except ConnectionAbortedError:
        print('Connection forcibly closed. Moving on to next round.')
    serverSocket.close()
    print("\nConnected client id:", connected_clients)
    return connected_clients


def handle_client(sock, bufsize, sep, path, connected_clients):
    try:
        received = sock.recv(bufsize).decode()
    except ConnectionAbortedError:
        print('Connection Lost')
    if sep in received:
        try: 
            clnt_id, filename, filesize = received.split(sep)
            try: 
                clnt_id = int(clnt_id)
            except ValueError:
                print('Client ID not a digit')
            filename = os.path.basename(filename)
            filesize = int(filesize)
            with open(path+"clnt_models/"+filename, "wb") as f:
                while True:
                    bytes_read = sock.recv(bufsize)
                    if not bytes_read: break
                    f.write(bytes_read)
            connected_clients.append(clnt_id)  # 클라이언트 ID 추가
        except FileNotFoundError:
            print('File Not Found. Might be the wrong path')
        except ConnectionAbortedError:
            print('Connection Lost')
    sock.close()

def recv_local_mdls(host, port, bufsize, sep, time_limit, path, max_clients):
    print("\n >>>>>>>   Receiving client models ...   <<<<<<<")
    connected_clients = []
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection_list = [serverSocket]
    serverSocket.bind((host, port))
    serverSocket.listen(max_clients)
    start = time.time()
    
    while time.time() - start < time_limit:
        read_socket, _, _ = select.select(connection_list, [], [], time_limit)
        for sock in read_socket:
            # New Connections
            if sock == serverSocket:
                clientSocket, _ = serverSocket.accept()
                connection_list.append(clientSocket)
            
            # Connected Before
            else:
                t = threading.Thread(target=handle_client, args=(sock, bufsize, sep, path, connected_clients))
                t.start()
                # End Communication
                connection_list.remove(sock)
    
    for t in threading.enumerate():
        if t != threading.current_thread():
            t.join()
    
    serverSocket.close()
    print("\n>>> Received Client models from :", connected_clients, f', {len(set(connected_clients))} clients')
    return connected_clients

