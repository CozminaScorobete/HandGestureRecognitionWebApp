import mysql.connector
##################################################################################################
#####DATABASE CONNECTION

def conection():
    connection=mysql.connector.connect(host="localhost", user="root", password="", database="licenta")

    if connection.is_connected():
        print("Connected Successfully")
    else:
        print("Fail to connect")

    connection.close()

    connection=mysql.connector.connect(host="localhost", user="root", password="", database="licenta")
    cursor= connection.cursor()

    return cursor