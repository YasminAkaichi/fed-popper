import socket

def cli_prompt():
    banner = """
   _____               ____                             
  / ___/______   __   / __ \____  ____  ____  ___  _____
  \__ \/ ___/ | / /  / /_/ / __ \/ __ \/ __ \/ _ \/ ___/
 ___/ / /   | |/ /  / ____/ /_/ / /_/ / /_/ /  __/ /    
/____/_/    |___/  /_/    \____/ .___/ .___/\___/_/     
                              /_/   /_/                 

"""
    print(banner)

def initialisation():
    print("Please introduce ... ")
    global nb_client
    global path_dir
    nb_client = int(input("- the number of Popper clients: "))
    path_dir = input("- the path to background files: ")

def popper_initialisation():
    # to be done
    # read background
    # produce constraint set
    print("We now read background and produce constraints")
    
def popper_compute_hypothesis():
    # to be done
    # using ASP produce an hypothesis and return it
    return [ "h1(X,Y) :- b1(X), c1(Y).",
             "h2(X,Y) :- b2(X), c2(Y)."
           ]

def tell_hypothesis(client,hyp):
    nb_cl = len(hyp)
    str_nb_cl = str(nb_cl)
    msg = f"tell( prgmlen({str_nb_cl}) )"
    client.send(msg.encode("utf-8")[:1024])
    client.recv(1024)
    for i in range(0,nb_cl):
        print("in loop")
        str_i = str(i)
        clause = "{" + hyp[i] + "}"
        print(f"clause = {clause}")
        msg = f"tell( prgm({str_i},{clause}) )"
        client.send(msg.encode("utf-8")[:1024])
        client.recv(1024)

def get_epsilon_pairs(client):
    global nb_client
    lepairs = []
    str_nb_client = str(nb_client)
    print(f"nb_client = {str_nb_client}")
    for i in range(1,nb_client+1):
        str_i = str(i)
        msg = f"ask( epair({str_i}) )"
        client.send(msg.encode("utf-8")[:1024])
        response = client.recv(1024)
        response = response.decode("utf-8")        
        lepairs.append(response)
    # msg = "reset"
    # client.send(msg.encode("utf-8")[:1024])
    # client.recv(1024)
    return lepairs

def popper_aggregate_epairs(lep):
    # to be done
    # here we return the pair (all,all)
    return "all", "all"

def popper_update_constraints(ep,em):
    # to be refined
    pass

def check_finish():
    # to be changed completely
    endi = input("Indicate if finish (0 = no, 1 = yes): ")
    return (endi == "1")
    
def run_client():
    # create a socket object
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # establish connection with server
    server_ip = "127.0.0.1"  # replace with the server's IP address
    server_port = 8000  # replace with the server's port number
    client.connect((server_ip, server_port))

    finish_learning = False
    hypothesis = []
    
    try:
        cli_prompt()
        initialisation()
        
        while not finish_learning:

            hypothesis = popper_compute_hypothesis()
            print(hypothesis)
            tell_hypothesis(client,hypothesis)
            lepairs = get_epsilon_pairs(client)
            eplus, eminus = popper_aggregate_epairs(lepairs)
            popper_update_constraints(eplus,eminus)
            finish_learning = check_finish()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # close client socket (connection to the server)
        client.close()
        print("Connection to server closed")


        
nb_client = 0
path_dir = " "
run_client()

