from typing import List, Tuple, Set
from subprocess import getoutput
import math
import time
from collections import defaultdict
from functools import reduce
from read_data import *
import numpy as np

cname, sname, qos, qos_lim = None, None, None, None
time_label = None
client_demand = None
bandwidth = None
start_time = None
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
        self.init_dispatch_again()
        if LOCAL: self.f = open('output/solution.txt', 'w')
        else: self.f = open('/output/solution.txt', 'w')
    
    def init_dispatch_again(self):
        self.record2 = None
        self.visited = set()  # element: (t_idx, s_idx, c_idx)
    
    def init_95(self):
        self.time_len = len(time_label)
        num_95 = math.ceil(self.time_len * 0.95)
        self.idx_95 = num_95 - 1
        self.higher_95_num = self.time_len - num_95
        self.server_5_t_idx = [ set() for _ in range(len(sname)) ]
        self.server_5_value = [ defaultdict(int) for _ in range(len(sname)) ]
    
    def qos_avail_for_c(self, c_idx: int) -> List[int]:
        c_qos = qos[:, c_idx]
        qos_avail = c_qos < qos_lim
        out = [ s_idx for s_idx, avail in enumerate(qos_avail) if avail ]
        return out
    
    def qos_avail_for_s(self, s_idx: int) -> List[int]:
        s_qos = qos[s_idx, :]
        qos_avail = s_qos < qos_lim
        out = [ c_idx for c_idx, avail in enumerate(qos_avail) if avail ]
        return out
    
    def check_output_valid(self):
        # check client is equal
        demand_sum = self.record.sum(axis=1)
        for t_idx, sum_at_each_time in enumerate(demand_sum):
            c_demand_at_t = client_demand[t_idx]
            if np.any(c_demand_at_t - sum_at_each_time):
            # if c_demand_at_t != sum_at_each_time:
                print(f'client demand is not equal at time {t_idx}')
                print(f'calculated: \n{sum_at_each_time} \n\n required: \n{c_demand_at_t}')
                print(f'difference (calculated_demand - required_demand): \n {sum_at_each_time - c_demand_at_t}')
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
        if LOCAL: self.info_print()
    
    def info_print(self):
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
    
    def assign(self, t_idx: int, s_idx: int, c_idx: int, demand: int) -> int: # has value: assign successfully  False: fail, need second time assign
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
            s_list = self.qos_avail_for_c(c_idx)
            occu_5_num = []
            occu_5_num = [ len(self.server_5_t_idx[s_idx])-(t_idx in self.server_5_t_idx[s_idx]) for s_idx in s_list ]
            arg = np.argsort(np.array(occu_5_num))
            s_arr = np.array(s_list)[arg]
            for idx, s_idx in enumerate(s_arr):
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
                    if self.server_5_value[s_idx][t_idx] == bandwidth[s_idx]: # server is full at current time, next loop # TODO: May not need it, delete
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

    def get_max_95_idx(self, array: np.ndarray):  # time * server
        value_at_95_list = []
        t_idx_list = []
        max_idx = self.higher_95_num + 1
        for time_series in array.T:
            t_idx = np.argpartition(time_series, -max_idx)[-max_idx]
            t_idx_list.append(t_idx)
            value_at_95_list.append(time_series[t_idx])
        value_arr = np.array(value_at_95_list)
        time_arr = np.array(t_idx_list)
        s_idx = np.argmax(value_arr)
        t_idx = time_arr[s_idx]
        return (t_idx, s_idx), value_arr[s_idx]

    def is_in_right_5(self, t_idx: int, s_idx: int, c_idx: int) -> bool:
        arr = self.record[:, s_idx, :].sum(axis=-1)
        # arr = self.record2[:, s_idx, c_idx].copy()
        i = 0
        while i < self.higher_95_num:
            cand = np.argmax(arr)
            if cand == t_idx: 
                return True
            arr[cand] = 0
            i += 1
        return False

    def how_much_can_add(self, t_idx: int, s_idx: int, c_idx: int):
        lower_than_95_ratio = 0.1
    
    def get_batch_max_95(self, arr: np.ndarray):
        value_at_95_list = []
        t_idx_95_list = []
        value_at_94_list = []
        t_idx_94_list = []
        max_idx = self.higher_95_num + 1
        next_to_max_idx = max_idx + 1
        for time_series in arr.T:
            t_idx = np.argpartition(time_series, -max_idx)[-max_idx]
            t_idx_95_list.append(t_idx)
            value_at_95_list.append(time_series[t_idx])  # 95 value for each server
            t_idx = np.argpartition(time_series, -next_to_max_idx)[-next_to_max_idx]
            t_idx_94_list.append(t_idx)
            value_at_94_list.append(time_series[t_idx])
        barrier_list = [ 0 for _ in range(len(sname)) ]
        out_t_idx_list = []
        out_s_idx_list = []
        out_res_at_95 = []
        value_95 = np.array(value_at_95_list)  # 95 value for each server
        idx_for_95 = np.argsort(np.array(value_at_95_list))[::-1]  # index for 95 value (sorting)
        ini_val = value_95[idx_for_95[0]]
        value_94 = np.array(value_at_94_list)
        for i in range(len(sname)):
            s_idx = idx_for_95[i]
            curr_val = value_95[s_idx]
            if curr_val == 0: break
            if ini_val / curr_val > 20: break
            out_res_at_95.append(curr_val)
            t_idx = t_idx_95_list[s_idx]
            out_t_idx_list.append(t_idx)
            out_s_idx_list.append(s_idx)
            barrier_list[s_idx] = value_94[s_idx]
        for j in range(i, len(sname)):
            s_idx = idx_for_95[i]
            barrier_list[s_idx] = value_95[s_idx]
        return (out_t_idx_list, out_s_idx_list, out_res_at_95), barrier_list
    
    def dispatch_to_small(self, barrier_list: List[int], t_idx: int, s_idx: int):
        barrier = barrier_list[s_idx]
        sum_at_here = self.record[t_idx, s_idx].sum()
        return max(barrier - sum_at_here - 1, 0)

    def dispatch_again(self):
        # if self.record2 is None:
        #     self.record2 = self.record.copy()
        # server_t_series = self.record2.sum(axis=-1)
        server_t_series = self.record.sum(axis=-1)

        # move to prev 95%
        (t_idx_list, s_idx_list, res_at_95_list), barrier_list = self.get_batch_max_95(server_t_series) # barrier is for each server
        # added_obj = set()
        added_obj = {}
        # for t_idx, s_idx in zip(t_idx_list, s_idx_list):
        #     # added_obj.add((t_idx, s_idx))
        #     added_obj[(t_idx, s_idx)] = 0
        for t_idx, s_idx_orig, res_at_95 in zip(t_idx_list, s_idx_list, res_at_95_list):
            client_series = self.record[t_idx, s_idx_orig]
            for c_idx, res in enumerate(client_series):
                if res > np.ceil(res_at_95 * 0.15):
                    demand = np.ceil(self.record[t_idx, s_idx_orig, c_idx] * 0.3).astype('int32')
                    s_idx_cand_list = self.qos_avail_for_c(c_idx)  # server candidate
                    for s_idx_new in s_idx_cand_list:
                        if (t_idx, s_idx_new) in added_obj:
                            dispatch_minus = added_obj[(t_idx, s_idx_new)]
                        else:
                            dispatch_minus = 0
                        if s_idx_new == s_idx_orig: continue
                        can_dispatch = self.dispatch_to_small(barrier_list, t_idx, s_idx_new)
                        if can_dispatch > dispatch_minus:
                            assign_bw = can_dispatch - dispatch_minus
                            self.assign(t_idx, s_idx_new, c_idx, assign_bw)
                            self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                            added_obj[(t_idx, s_idx_new)] = can_dispatch



        # (t_idx, s_idx), res_at_95 = self.get_max_95_idx(server_t_series)
        # # prevent visit 2nd time
        # self.record2[t_idx, s_idx] = 0
        # # dispatch for these moved client
        # client_series = self.record[t_idx, s_idx]
        # for c_idx, res in enumerate(client_series):
        #     if res > np.ceil(res_at_95 * 0.15): # client ratio larger than 15% at 95%
        #         demand = np.ceil(self.record[t_idx, s_idx, c_idx] * 0.3).astype('int32')
        #         s_idx_list = self.qos_avail_for_c(c_idx)
        #         # in right 5%
        #         for new_s_idx in s_idx_list:
        #             if new_s_idx == s_idx: continue
        #             if self.is_in_right_5(t_idx, new_s_idx, c_idx):
        #                 left = self.assign(t_idx, new_s_idx, c_idx, demand)
        #                 if left:
        #                     assign_bw = demand - left
        #                 else:
        #                     assign_bw = demand
        #                 if np.any(client_demand[0] - self.record[0].sum(axis=0)): # TODO: delete
        #                     a=1
        #                 demand = left
        #                 self.record[t_idx, s_idx, c_idx] -= assign_bw
        #                 # self.record[t_idx, new_s_idx, c_idx] += assign_bw
        #                 self.record2[t_idx, s_idx, c_idx] -= assign_bw
        #                 self.record2[t_idx, new_s_idx, c_idx] += assign_bw
        #                 if demand == 0: break
        #         # uncomment it
        #         # if demand == 0: continue

        #         # TODO: to be continue
        #         # # not in right 5%
        #         # for new_s_idx in s_idx_list:
        #         #     if new_s_idx == s_idx: continue
        #         #     # add to left 95% will not exceed 95%
        #         #     # self.how_much_can_add(t_idx, s_idx, c_idx)
        #         #     if demand == 0: break
        
        
        # # traffic comes from these client, search where to put these traffic


if __name__ == '__main__':
    start_time = time.time()
    get_data()
    s = Solution()
    s.dispatch()
    # s.output()
    s.info_print()
    if LOCAL: s.check_output_valid()
    # second time dispatch
    if LOCAL: time_threshould = 30
    else: time_threshould = 280
    while time.time() - start_time < time_threshould:
        s.dispatch_again()
    s.output()
    if LOCAL: s.check_output_valid()