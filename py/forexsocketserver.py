import socket
import ast
import sys
ocp-
import build
import json
import time


class socketserver:
    def __init__(self, address='', port=9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''

    def recv_timeout(self, timeout=2):
        print('Set the socket non blocking')
        self.conn.setblocking(False)

        print('Initialize total_data and data ...')
        # total data partwise in an array
        total_data = []
        data = ''

        # beginning time
        begin = time.time()
        print("Set begin time: %d " % begin)
        while True:
            # if you got some data, then break after timeout
            if total_data and time.time() - begin > timeout:
                print('Got some data, then break after timeout')
                break

            # if you got no data at all, wait a little longer, twice the timeout
            elif time.time() - begin > timeout * 2:
                print('Got no data at all, wait a little longer, twice the timeout')
                break

            # recv something
            try:
                data = self.conn.recv(8192)
                print('recv %d' % len(data))
                if data:
                    total_data.append(data)
                    # change the beginning time for measurement
                    begin = time.time()
                else:
                    # sleep for sometime to indicate a gap
                    print('sleep for sometime to indicate a gap')
                    time.sleep(0.1)
            except socket.timeout as e:
                err = e.args[0]
                # this next if/else is a bit redundant, but illustrates how the
                # timeout exception is setup
                if err == 'timed out':
                    time.sleep(1)
                    print('recv timed out, retry later')
                    continue
                else:
                    print(e)
                    sys.exit(1)
            except socket.error as e:
                # Something else happened, handle error, exit, etc.
                print('Something else happened')
                print(e)
                pass

        # join all parts to make final string
        print('Return total_data %d' % len(total_data))
        return b''.join(total_data).decode("utf-8")

    def recv_block(self):
        print('Initialize total_data and data ...')
        # total data partwise in an array
        total_data = []
        data = ''

        while True:
            # recv something
            try:
                data = self.conn.recv(8192)
                print('recv %d' % len(data))
                if data:
                    total_data.append(data)
                else:
                    print('No data')
                    break
            except socket.timeout as e:
                err = e.args[0]
                # this next if/else is a bit redundant, but illustrates how the
                # timeout exception is setup
                if err == 'timed out':
                    time.sleep(1)
                    print('recv timed out, retry later')
                    continue
                else:
                    print(e)
                    break
            except socket.error as e:
                # Something else happened, handle error, exit, etc.
                print('Some error happened')
                print(e)
                break
            except:
                print('Something else happened')
                break

            # join all parts to make final string
        print('Return total_data %d' % len(total_data))
        return b''.join(total_data).decode("utf-8")

    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        self.cummdata = self.recv_timeout()
        print('send cummdata: %s' % self.cummdata)
        self.conn.send(bytes(build.train_test_model(self.cummdata), "utf-8"))
        return self.cummdata

    def __del__(self):
        self.sock.close()


serv = socketserver('0.0.0.0', 9090)

print('Socket Created at {}. Waiting for client..'.format(serv.sock.getsockname()))

while True:
    msg = serv.recvmsg()
