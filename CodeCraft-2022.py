from typing import List, Tuple, Set
from subprocess import getoutput
import math
from read_data import *
import numpy as np
from solution1 import Solution

cname, sname, qos, qos_lim = None, None, None, None
time_label = None
client_demand = None
bandwidth = None
LOCAL = getoutput('uname') == 'Darwin'


class Solution():
    def __init__(self) -> None:
        self.burden_ratio = 0.5
        self.top_n_common_qos_ratio = 0.3
        self.top_n_c_idx: List[int] = None
        self.qos_avail_server_idx: List[int] = None
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
    
    def _top_n_common_qos_avail(self, demand: np.ndarray) -> Set[int]:
        top_n = math.ceil(self.top_n_common_qos_ratio * len(cname))
        sep_pos = len(demand) - top_n
        arg = np.argpartition(demand, sep_pos)
        arg = arg[sep_pos: ]
        self.top_n_c_idx = arg
        qos_avail = np.ones(len(sname), dtype=bool)
        for c_idx in arg:
            c_qos = qos[:, c_idx]
            c_qos = c_qos < qos_lim
            qos_avail = qos_avail & c_qos
        qos_avail_server_idx = [ i for i, avail in enumerate(qos_avail) if avail ]
        self.qos_avail_server_idx = set(qos_avail_server_idx)
    
    def try_assign(self, s_idx: int, demand: int) -> int:
        if self.server_remain[s_idx] >= demand:
            return 0
        return demand - self.server_remain[s_idx]
    
    def assign(self, c_idx: int, s_idx: int, res: int):
        self.server_remain[s_idx] -= res
        for r_idx, r in enumerate(self.record[c_idx]):
            if s_idx == r[0]:
                self.record[c_idx][r_idx] = (s_idx, res + r[1])
                return
        self.record[c_idx].append((s_idx, res))

    def dispatch(self):
        for idx, each in enumerate(client_demand):
            self._top_n_common_qos_avail(each)
            for c_idx, c_demand in enumerate(each):
                self.dispatch_for_client(c_idx, c_demand)
            self.output()  # TODO: merge multiple dispatch
    
    def dispatch_for_client(self, c_idx, demand: int):
        is_top_n_c_idx = c_idx in self.top_n_c_idx
        if is_top_n_c_idx:
            ratio = 1 / (1+math.exp(-0.01*len(self.qos_avail_server_idx))) - 0.2
            other_burden = math.ceil(demand * ratio)
            # other_burden = math.ceil(demand * self.burden_ratio)
            main_demand = demand - other_burden
        else:
            main_demand = demand
        c_qos = qos[:, c_idx]
        qos_avail = c_qos < qos_lim
        for s_idx, avail in enumerate(qos_avail):
            if not avail: continue
            if is_top_n_c_idx:
                self.qos_avail_server_idx.discard(s_idx)
            assign_left = self.try_assign(s_idx, main_demand)
            if assign_left:
                self.assign(c_idx, s_idx, main_demand - assign_left)
                main_demand = assign_left
            else:
                self.assign(c_idx, s_idx, main_demand)
                break 
        # TODO: can be better
        # dispatch for other0 burden
        if is_top_n_c_idx:
            qos_avail_num = len(self.qos_avail_server_idx)
            burden_dispatched_num = -1
            for s_idx in self.qos_avail_server_idx:
                if other_burden == 0:
                    break
                burden_dispatched_num += 1
                div_num = qos_avail_num - burden_dispatched_num
                assign_res = math.ceil(other_burden / div_num)
                assign_left = self.try_assign(s_idx, assign_res)
                if assign_left:
                    self.assign(c_idx, s_idx, assign_res - assign_left)
                    other_burden = assign_left + ( other_burden - assign_res )
                else:
                    self.assign(c_idx, s_idx, assign_res)
                    other_burden -= assign_res
        # TODO: may have bug, not fill c_idx


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

def main():
    get_data()
    s = Solution()
    s.dispatch()


if __name__ == '__main__':
    main()
    