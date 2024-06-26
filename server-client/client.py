import socket

def Main():
    host = '127.0.0.1'
    port = 12345

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print("Hello, I’m the Oracle. How can I help you today?")
    while True:
        message = input("Enter your weather query (e.g., 'What is London's weather(Or percipitation .etc) on 29 March 2024?' or type 'exit' to quit): ")
        if message.lower() == 'exit':
            break

        s.send(message.encode('utf-8'))
        data = s.recv(1024).decode('utf-8')
        print('Received from the server:')
        print(data)

    s.close()

if __name__ == '__main__':
    Main()