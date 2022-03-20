from typing import List, Tuple
from subprocess import getoutput

LOCAL = getoutput('uname') == 'Darwin'

def read_demand():
    fname = 'data/demand.csv'
    if not LOCAL: fname = '/' + fname
    f = open(fname, 'r'); data = f.readlines(); f.close()
    client_name = data[0].strip().split(',')[1:]
    client_demand = []; time_label = []
    for each in data[1:]:
        d = each.strip().split(',')
        time_label.append(d[0])
        client_demand.append(list(map(int, d[1:])))
    return time_label, client_name, client_demand

def read_server_bandwidth():
    fname = 'data/site_bandwidth.csv'
    if not LOCAL: fname = '/' + fname
    server_name = []; server_bandwidth = []
    f = open(fname, 'r'); data = f.readlines(); f.close()
    for each in data[1:]:
        sname, bw4server = each.strip().split(',')
        server_name.append(sname)
        server_bandwidth.append(int(bw4server))
    return server_name, server_bandwidth

def read_qos():
    fname = 'data/qos.csv'
    if not LOCAL: fname = '/' + fname
    f = open(fname, 'r'); data = f.readlines(); f.close()
    cname = data[0].strip().split(',')[1:]
    sname = []; qos_array4server = []
    for each in data[1:]:
        qos_line_split = each.strip().split(',')
        sname.append(qos_line_split[0])
        qos_array4server.append(list(map(int, qos_line_split[1:])))
    return cname, sname, qos_array4server

def read_qos_limit():
    fname = 'data/config.ini'
    if not LOCAL: fname = '/' + fname
    f = open(fname, 'r'); data = f.readlines(); f.close()
    qos_lim = int(data[1].strip().split('=')[1])
    return qos_lim