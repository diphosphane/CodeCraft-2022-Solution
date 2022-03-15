from typing import List, Tuple
from subprocess import getoutput

LOCAL = getoutput('uname') == 'Darwin'

def read_demand() -> Tuple[List[str], List[int]]:
    fname = 'data/demand.csv'
    if not LOCAL:
        fname = '/' + fname
    with open(fname) as f:
        data = f.read().splitlines()
    client_name = data[0].split(',')[1:]
    client_demand = []
    time_label = []
    for each in data[1:]:
        d = each.split(',')
        time_label.append(d[0])
        client_demand.append(list(map(int, d[1:])))
    return time_label, client_name, client_demand

def read_server_bandwidth() -> Tuple[List[str], List[int]]:
    fname = 'data/site_bandwidth.csv'
    if not LOCAL:
        fname = '/' + fname
    with open(fname) as f:
        data = f.read().splitlines()
    server_name = []
    server_bandwidth = []
    for each in data[1:]:
        a, b = each.split(',')
        server_name.append(a)
        server_bandwidth.append(int(b))
    return server_name, server_bandwidth

def read_qos() -> Tuple[List[str], List[str], List[List[int]]]:
    fname = 'data/qos.csv'
    if not LOCAL:
        fname = '/' + fname
    with open(fname) as f:
        data = f.read().splitlines()
    client_name = data[0].split(',')[1:]
    server_name = []
    qos_array = []
    for each in data[1:]:
        d = each.split(',')
        server_name.append(d[0])
        qos_array.append(list(map(int, d[1:])))
    return client_name, server_name, qos_array

def read_qos_limit() -> int:
    fname = 'data/config.ini'
    if not LOCAL:
        fname = '/' + fname
    with open(fname) as f:
        data = f.read().splitlines()
    qos_lim = int(data[1].split('=')[1])
    return qos_lim