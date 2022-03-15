from typing import List, Tuple, Set
from subprocess import getoutput
from read_data import *

# global cname, sname, qos, qos_lim, time_label, client_demand, bandwidth
LOCAL = getoutput('uname') == 'Darwin'

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
