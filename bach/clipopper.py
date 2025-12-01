import socket
from parser import Parser
from data_structures import (
    Abs_SITerm,
    SI_ATOMIC, 
    SI_COMPOSED,
    SI_HORN_CLAUSE,
    SI_PRGM,
    SI_THEORIES, 
    AST_PRIMITIVE,
    AST_CLOSE_FUNCTION,
)


def cli_prompt():
    banner = """

 __ .   .__                
/  `|*  [__) _ ._ ._  _ ._.
\__.||  |   (_)[_)[_)(/,[  
               |  |        

"""
    print(banner)


def initialisation():
    print("Please introduce ... ")
    global str_id_nb
    global path_dir
    str_id_nb = input("- the number to identify the client: ")
    path_dir = input("- the path to example files: ")

def get_nb_clause_from_prgmlen_si(ast):
    try:
        arg_prgmlen_si = ast.arguments
        nb_cl = arg_prgmlen_si[0]
        return nb_cl
    except Exception as e:
        print(f"Error: {e}")

def popper_read_hypothesis(client):
    msg = f"ask( prgmlen )"
    client.send(msg.encode("utf-8")[:1024])
    response = client.recv(1024)
    response = response.decode("utf-8")
    print(response)
    ast = mycliparser.parse_comAugStInfo(response)
    nb_cl = get_nb_clause_from_prgmlen_si(ast)
    print("nb_cl = ")
    print(str(nb_cl))
    return []

    
def popper_test_hypothesis(hyp):
    return "all", "all"

def popper_report_epair(client,id_cli,eplus,eminus):
    msg = f"tell( epair({id_cli},{eplus},{eminus}) )"
    client.send(msg.encode("utf-8")[:1024])
    response = client.recv(1024)
    response = response.decode("utf-8")
    
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

            hypothesis = popper_read_hypothesis(client)
            print(hypothesis)
            eplus, eminus = popper_test_hypothesis(hypothesis)
            popper_report_epair(client,str_id_nb,eplus,eminus)
            finish_learning= check_finish()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # close client socket (connection to the server)
        client.close()
        print("Connection to server closed")

mycliparser = Parser()        
str_id_nb = "0"
path_dir = " "
run_client()

