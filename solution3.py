from pickletools import read_int4
from typing import List, Tuple, Set
from subprocess import getoutput
import math
from collections import defaultdict
from functools import reduce
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

class Solution():
    def __init__(self) -> None:
        self.record = np.zeros((len(time_label), len(sname), len(cname)), dtype=np.int32)
        self.init_95()
        if LOCAL: self.f = open('output/solution.txt', 'w')
        else: self.f = open('/output/solution.txt', 'w')
    
    def init_95(self):
        self.time_len = len(time_label)
        num_95 = math.ceil(self.time_len * 0.95)
        self.idx_95 = num_95 - 1
        self.higher_95_num = self.time_len - num_95
        self.server_5_t_idx = [ set() for _ in range(len(sname)) ]
        self.server_5_value = [ defaultdict(int) for _ in range(len(sname)) ]
    
    def qos_avail(self, c_idx: int) -> List[int]:
        c_qos = qos[:, c_idx]
        qos_avail = c_qos < qos_lim
        out = [ s_idx for s_idx, avail in enumerate(qos_avail) if avail ]
        return out

    # def __del__(self):
    #     self.f.close()
    
    def check_output_valid(self):
        # check client is equal
        demand_sum = self.record.sum(axis=1)
        for t_idx, sum_at_each_time in enumerate(demand_sum):
            c_demand_at_t = client_demand[t_idx]
            if np.any(c_demand_at_t - sum_at_each_time):
            # if c_demand_at_t != sum_at_each_time:
                print(f'client demand is not equal at time {t_idx}')
                print(f'calculated: \n{sum_at_each_time} \n\n required: \n{c_demand_at_t}')
                exit(1)
        if np.any(demand_sum - client_demand):
        # if demand_sum != client_demand:
            print('client demand is not equal')
            exit(1)
        # check qos
        for t_idx, r_each_time in enumerate(self.record):
            for s_idx, r_each_s in enumerate(r_each_time):
                for c_idx, value in enumerate(r_each_s):
                    if value:
                        if qos[s_idx, c_idx] > qos_lim:
                            print(f'qos not satisfied in time {t_idx}, server {sname[s_idx]}, client {cname[c_idx]}')
                            exit(1)
        # check server upper limit
        bw_sum = self.record.sum(axis=-1)
        for t_idx, sum_at_t in enumerate(bw_sum):
            if np.any(sum_at_t > bandwidth):
                print(f'exceed bandwidth upper at time {t_idx} {time_label[t_idx]}')
                # print(f'solution sum: \n{sum_at_t} \n\n bandwidth limit: \n{bandwidth}')
                print(f'different (bandwidth_limit - solution_sum): \n{bandwidth - sum_at_t}')
                exit(1)
        print('test passed \n\n\n')

    def output(self):
        for each_time_step_operation in self.record:
            for c_idx, s_series in enumerate(each_time_step_operation.T):
                tmp = cname[c_idx] + ':'
                out_list = []
                for s_idx, res in enumerate(s_series):
                    if res:
                        out_list.append(f'<{sname[s_idx]},{res}>')
                tmp += ','.join(out_list)
                self.f.write(tmp + '\n')
        self.f.close()
        # calc score 
        if LOCAL:
            bd_each_time = self.record.sum(axis=-1)
            bd_each_time.sort(axis=0)
            score_95 = bd_each_time[self.idx_95, :]
            final_score = score_95.sum()
            print(f'95% score sum: {final_score}\n{score_95}\n')
    
    @staticmethod
    def get_max_idx(array: np.ndarray) -> Tuple[int, int]:
        arr = array.copy()
        cnt = 0
        while cnt < reduce(lambda x,y: x*y, arr.shape):
            idx = np.unravel_index(np.argmax(arr), arr.shape)
            yield idx, arr[idx]
            arr[idx] = 0
            cnt += 1
    
    def assign(self, t_idx: int, s_idx: int, c_idx: int, demand: int) -> bool: # has value: assign successfully  False: fail, need second time assign
        add_up = self.record[t_idx, s_idx].sum() + demand
        upper_limit = bandwidth[s_idx]
        if add_up > upper_limit: # assign fail
            left = add_up - upper_limit
            assign_bandwidth = demand - left
            self.record[t_idx, s_idx, c_idx] += assign_bandwidth
            return left
        self.record[t_idx, s_idx, c_idx] += demand
        return 0

    def dispatch(self):
        for (t_idx, c_idx), demand in self.get_max_idx(client_demand):
            s_list = self.qos_avail(c_idx)
            occu_5_num = []
            occu_5_num = [ len(self.server_5_t_idx[s_idx])-(t_idx in self.server_5_t_idx[s_idx]) for s_idx in s_list ]
            arg = np.argsort(np.array(occu_5_num))
            s_arr = np.array(s_list)[arg]
            for idx, s_idx in enumerate(s_arr):
                if s_idx == 90 and t_idx == 72:
                    a = 1
                if t_idx in self.server_5_t_idx[s_idx]: # in server top 5, put all the resources into
                    if self.server_5_value[s_idx][t_idx] == bandwidth[s_idx]: # server is full at current time, next loop
                        continue
                    else: # server is not full, try to fill it to full
                        left = self.assign(t_idx, s_idx, c_idx, demand)
                        if left:
                            assign_bandwidth = demand - left
                            self.server_5_value[s_idx][t_idx] += assign_bandwidth
                            demand = left
                            continue
                        else:
                            self.server_5_value[s_idx][t_idx] += demand
                            demand = 0
                            break
                elif len(self.server_5_t_idx[s_idx]) != self.higher_95_num: # not in server top 5, top 5 is not full, fill a blank
                    if self.server_5_value[s_idx][t_idx] == bandwidth[s_idx]: # server is full at current time, next loop
                        continue
                    self.server_5_t_idx[s_idx].add(t_idx)
                    left = self.assign(t_idx, s_idx, c_idx, demand)
                    if left:
                        assign_bandwidth = demand - left
                        self.server_5_value[s_idx][t_idx] = assign_bandwidth
                        demand = left
                        continue
                    else:
                        self.server_5_value[s_idx][t_idx] = demand
                        demand = 0
                        break
                else:  # not in top 5, top 5 is full, put average in all the avail
                    avg_s_arr = s_arr[idx:]
                    avg_dispatch = math.floor(demand / len(avg_s_arr))
                    remain = demand - avg_dispatch * len(avg_s_arr)
                    for ss_idx in avg_s_arr:
                        left = self.assign(t_idx, ss_idx, c_idx, avg_dispatch + remain)
                        if left:
                            remain = left
                            continue
                        else: remain = 0
                    if remain: raise BaseException("dispatch fail, has remain")
                    demand = 0
                    break
            if demand:
                raise BaseException("dispatch fail")


if __name__ == '__main__':
    get_data()
    s = Solution()
    s.dispatch()
    s.output()
    if LOCAL: s.check_output_valid()