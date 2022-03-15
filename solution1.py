from typing import List, Tuple, Set
from subprocess import getoutput
import math
from read_data import *
import numpy as np

cname, sname, qos, qos_lim = None, None, None, None
time_label = None
client_demand = None
bandwidth = None
LOCAL = getoutput('uname') == 'Darwin'

def get_data():
    global cname, sname, qos, qos_lim, bandwidth, client_demand, time_label
    cname, sname, qos = read_qos()
    qos = np.array(qos)
    time_label, client_name, client_demand = read_demand()
    client_idx_list = []
    for c in cname:
        idx = client_name.index(c)
        client_idx_list.append(idx)
    client_demand = np.array(client_demand)[:, client_idx_list]
    server_name, server_bandwidth = read_server_bandwidth()
    bandwidth = []
    for s in sname:
        idx = server_name.index(s)
        bandwidth.append(server_bandwidth[idx])
    qos_lim = read_qos_limit()
    bandwidth = np.array(bandwidth)

class Solution:
    def __init__(self) -> None:
        self.reset()
        if LOCAL:
            self.f = open('output/solution.txt', 'w')
        else:
            self.f = open('/output/solution.txt', 'w')

    def __del__(self):
        self.f.close()

    def reset(self):
        self.record: List[List[Tuple[int, int]]] = [ [] for _ in range(len(cname)) ]
        self.server_remain: List[int] = bandwidth.copy()
    
    def try_assign(self, s_idx: int, demand: int) -> int:
        if self.server_remain[s_idx] >= demand:
            return 0
        return demand - self.server_remain[s_idx]
    
    def assign(self, c_idx: int, s_idx: int, res: int):
        self.server_remain[s_idx] -= res
        self.record[c_idx].append((s_idx, res))
    
    def output(self):
        for c_idx, r in enumerate(self.record):
            tmp = cname[c_idx] + ':'
            if r:
                out_list = []
                for s_idx, res in r:
                    out_list.append(f'<{sname[s_idx]},{res}>')
                tmp += ','.join(out_list)
            self.f.write(tmp + '\n')
        self.reset()
    

    def dispatch_for_client(self, c_idx: int, demand: int):
        c_qos = qos[:, c_idx]
        qos_avail = c_qos < qos_lim
        for s_idx, avail in enumerate(qos_avail):
            if not avail: continue
            assign_left = self.try_assign(s_idx, demand)
            if assign_left:
                self.assign(c_idx, s_idx, demand - assign_left)
                demand = assign_left
            else:
                self.assign(c_idx, s_idx, demand)
                return 
    
    def dispatch(self):
        for each in client_demand:
            for c_idx, c_demand in enumerate(each):
                self.dispatch_for_client(c_idx, c_demand)
            self.output()

if __name__ == '__main__':
    get_data()
    s = Solution()
    s.dispatch()