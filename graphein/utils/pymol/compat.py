try:
    from xmlrpc.client import ServerProxy as Server
except BaseException:
    from xmlrpclib import Server
