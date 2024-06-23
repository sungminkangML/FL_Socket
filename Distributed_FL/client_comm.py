from utils_libs import *

def recv_global_mdl(host, port, bufsize, sep, clnt_id, path):
    # Creating Socket
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.settimeout(10)

    # Connecting with Server Socket
    try:
        clientSocket.connect((host, port))
        
        # Sending Client ID
        clientSocket.sendall(clnt_id.encode())
    except ConnectionRefusedError:
        print('Cannot connect to Server')
        print('Check the host and port number of the server')
        print('Check your client id and your file path')
    except ConnectionAbortedError:
        print('Connection Lost')
        
    # Receiving Current Round
    try:
        curr_round = clientSocket.recv(16).decode()
        curr_round = int(curr_round)
        print('\n---------------- Current round is:', curr_round, '----------------')
    except ValueError:
        print('Invalid value due to distracted communication')

    # Receiving Global Model
    try:
        received_global = clientSocket.recv(bufsize).decode()
        filename, filesize = received_global.split(sep)
        filename = os.path.basename(filename)
        filesize = int(filesize)

        with open(path+"global_models/"+filename, "wb") as f:
            while True:
                bytes_read = clientSocket.recv(bufsize)
                if not bytes_read: break
                f.write(bytes_read)
        if filesize:
            print("Successfully received global model from Server")
    except ConnectionRefusedError:
        print('Connection Lost. Participate in the next round.')
    except ValueError:
        print('Invalid value due to distracted communication')
    
    clientSocket.close()
    return curr_round


def send_local_mdl(host, port, bufsize, sep, clnt_id, round, path, msg):
    def send_model():
        if "y" in msg or "Y" in msg or "yes" in msg:
            print('Sending Local model to server ...')
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
                    
            # send local model
            filename = path + f"clnt_models/clnt{clnt_id}_mdl{round}.pt"
            filesize = os.path.getsize(filename)
            clientSocket.sendall(f"{clnt_id}{sep}{filename}{sep}{filesize}".encode())
            time.sleep(1)
            with open(filename, "rb") as f:
                while True:
                    bytes_read = f.read(bufsize)
                    if not bytes_read: break
                    clientSocket.sendall(bytes_read)
            clientSocket.close()
            print('Successfully sent local model to server\n\n')
        else:
            print('Skipping current round\n\n')

    t = threading.Thread(target=send_model)
    t.start()