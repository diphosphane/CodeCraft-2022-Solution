from collections import defaultdict
from typing import List, Tuple, Set
from subprocess import getoutput
import math
import time
from functools import reduce
from read_data import *
import numpy as np

cname, sname, qos, qos_lim = None, None, None, None
t_len, s_len, c_len = 0, 0, 0
time_label = None
client_demand = None
bandwidth = None
start_time = None
LOCAL = getoutput('uname') == 'Darwin'

def get_data():
    global cname, sname, qos, qos_lim, bandwidth, client_demand, time_label
    global t_len, s_len, c_len
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
    t_len, s_len, c_len = len(time_label), len(sname), len(cname)

class Solution():
    def __init__(self) -> None:
        self.init_95()
        self.init_qos()
        self.record = np.zeros((t_len, s_len, c_len), dtype=np.int32)
        self.t_s_record = np.zeros((t_len, s_len), dtype=np.int32)
        self.t_s_include_c = [ [ set() for _ in range(s_len) ] for _ in range(t_len) ]
    
    def init_qos(self):
        self.qos_avail_for_c = []
        for c_idx in range(c_len):
            self.qos_avail_for_c.append(self._qos_avail_for_c(c_idx))
        self.qos_avail_for_s = []
        for s_idx in range(s_len):
            self.qos_avail_for_s.append(self._qos_avail_for_s(s_idx))
        self.s2s_bridge = []
        for s_idx in range(s_len):
            s = set()
            for c_idx in self.qos_avail_for_s[s_idx]:
                s.update(self.qos_avail_for_c[c_idx])
            self.s2s_bridge.append(s)
    
    def init_95(self):
        num_95 = math.ceil(t_len * 0.95)
        self.idx_95 = num_95 - 1
        self.higher_95_num = t_len - num_95
        self.server_5_t_idx = [ set() for _ in range(s_len) ]
        self.server_5_value = [ defaultdict(int) for _ in range(s_len) ]
    
    def _qos_avail_for_c(self, c_idx: int) -> List[int]:
        c_qos = qos[:, c_idx]
        qos_avail = c_qos < qos_lim
        out = [ s_idx for s_idx, avail in enumerate(qos_avail) if avail ]
        return out
        # return set(out)
    
    def _qos_avail_for_s(self, s_idx: int) -> List[int]:
        s_qos = qos[s_idx, :]
        qos_avail = s_qos < qos_lim
        out = [ c_idx for c_idx, avail in enumerate(qos_avail) if avail ]
        return out
        # return set(out)
    
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
                            print(f'qos not satisfied in time {t_idx}, server {sname[s_idx]} (index: {s_idx}), client {cname[c_idx]} (index: {c_idx})')
                            exit(1)
                        if value < 0:
                            print(f'dispatch bandwidth < 0 in time {t_idx}, server {sname[s_idx]} (index: {s_idx}), client {cname[c_idx]} (index: {c_idx})')
                            exit(1)
        # check server upper limit
        # bw_sum = self.record.sum(axis=-1)
        bw_sum = self.t_s_record
        for t_idx, sum_at_t in enumerate(bw_sum):
            if np.any(sum_at_t > bandwidth):
                print(f'exceed bandwidth upper at time {t_idx} {time_label[t_idx]}')
                print(f'different (bandwidth_limit - solution_sum): \n{bandwidth - sum_at_t}')
                exit(1)
        print('test passed \n')

    def output(self):
        if LOCAL: self.f = open('output/solution.txt', 'w')
        else: self.f = open('/output/solution.txt', 'w')
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
        if LOCAL: self.calc_score95()
    
    def calc_score95(self, print_sep=True):
        # bw_each_time = self.record.sum(axis=-1)
        bw_each_time = self.t_s_record.copy()
        bw_each_time.sort(axis=0)
        score_95 = bw_each_time[self.idx_95, :]
        final_score = score_95.sum()
        if print_sep:
            print(f'95% score sum: {final_score}\n{sorted(score_95, reverse=True)}\n')
        else:
            print(f'95% score sum: {final_score}')
        return final_score
    
    @staticmethod
    def get_max_idx_gen(array: np.ndarray) -> Tuple[int, int]:
        arr = array.copy()
        cnt = 0
        while cnt < reduce(lambda x,y: x*y, arr.shape):
            idx = np.unravel_index(np.argmax(arr), arr.shape)
            yield idx, arr[idx]
            arr[idx] = 0
            cnt += 1
    
    def assign(self, t_idx: int, s_idx: int, c_idx: int, demand: int) -> Tuple[int, int]: # left, assigned
        # add_up = self.record[t_idx, s_idx].sum() + demand
        self.t_s_include_c[t_idx][s_idx].add(c_idx)
        add_up = self.t_s_record[t_idx, s_idx] + demand
        upper_limit = bandwidth[s_idx]
        if add_up > upper_limit: # assign fail
            left = add_up - upper_limit
            assign_bandwidth = demand - left
            self.record[t_idx, s_idx, c_idx] += assign_bandwidth
            self.t_s_record[t_idx, s_idx] += assign_bandwidth
            return left, assign_bandwidth
        self.record[t_idx, s_idx, c_idx] += demand
        self.t_s_record[t_idx, s_idx] += demand
        return 0, demand
    
    def _get_s_idx_arr(self, s_list: List[int], occu_num: List[int], t_idx: int):
        if True: # legacy
            arg = np.argsort(np.array(occu_num))
            s_arr = np.array(s_list)[arg]
        else:  # new way to dispatch, to more space
            s_arr = np.array(s_list)
            left_res = [ bandwidth[s_idx] - self.server_5_value[s_idx].get(t_idx, 0) for s_idx in s_list ]
            arg = np.lexsort((-np.array(left_res), np.array(occu_num)))
            s_arr = np.array(s_list)[arg]
        return s_arr
    
    def empty_analyse(self, input_record=None):
        if input_record is not None:
            record = input_record
        else: record = self.record
        pos_96 = np.ceil(t_len * 0.95 ).astype('int32')
        res_t_for_server = record.sum(axis=-1).T # s_idx, t_idx
        t_idx_arr_for_server = []
        for t_series in res_t_for_server:
            idxs = np.argpartition(t_series, pos_96)[pos_96:]
            t_idx_arr_for_server.append(idxs)
        idle_matrix = [] # s_idx, t_idx
        for s_idx, t_idx_arr in enumerate(t_idx_arr_for_server):
            used_bw = res_t_for_server[s_idx][t_idx_arr]
            upper_bw = bandwidth[s_idx]
            idle_bw = upper_bw - used_bw
            # idle_perc = idle_bw / upper_bw
            idle_matrix.append(idle_bw)
        idle_matrix = np.array(idle_matrix)
        idle_matrix_idx = np.array(t_idx_arr_for_server) # s_idx, t_idx
        return idle_matrix, idle_matrix_idx  # idle_value and its t_idx for each server
    
    def idx_of_max_idle(self, input_record=None):
        idle_matrix, idle_matrix_idx = self.empty_analyse(input_record)
        while True: # TODO: consider add max count
            max_idx = np.unravel_index(np.argmax(idle_matrix), idle_matrix.shape)
            value = idle_matrix[max_idx]
            t_idx = idle_matrix_idx[max_idx]
            s_idx, _ = max_idx
            if value <= 10: break
            yield s_idx, t_idx, value
            idle_matrix[max_idx] = 0
    
    def fill_idle_after_95(self):
        for i, (s_idx, t_idx, idle_value) in enumerate(self.idx_of_max_idle()):
            self.fill_one_idle_after_95(s_idx, t_idx, idle_value, 0)
            # if filled_bw == 0: continue
            # self.record[t_idx, s_idx, c_idx] += filled_bw
    
    def is_at_95(self, s_idx: int, t_idx: int) -> bool:
        # t_series = self.record[:, s_idx].sum(axis=-1)
        t_series = self.t_s_record[:, s_idx]
        t_idx_95 = np.argpartition(t_series, self.idx_95)[self.idx_95]
        return t_idx == t_idx_95
    
    def is_after_95(self, s_idx: int, t_idx: int) -> bool:
        # t_series = self.record[:, s_idx].sum(axis=-1)
        t_series = self.t_s_record[:, s_idx]
        t_idx_95_list = np.argpartition(t_series, self.idx_95)[(self.idx_95+1):]
        return t_idx in t_idx_95_list

    def fill_one_idle_after_95(self, s_idx: int, t_idx: int, idle_value: int, layer: int=0) -> bool: # success or not # -> Tuple[int, int]: # can_fill_bw, c_idx 
        if layer >= 1: return False
        # find the proper c_idx that can fill
        c_idx_list = self.qos_avail_for_s[s_idx]
        for c_idx in c_idx_list:
            new_s_idx_list = self.qos_avail_for_c[c_idx]
            for new_s_idx in new_s_idx_list:
                # at t_idx, new_s_idx can be 95%,  fill [t_idx, s_idx, c_idx]  remove [t_idx, new_s_idx, c_idx]
                if self.is_at_95(new_s_idx, t_idx):
                    new_cand_has_bw = self.record[t_idx, new_s_idx, c_idx]
                    print(f'can move to >95%, {new_cand_has_bw}')
                    self.t_s_include_c[t_idx][s_idx].add(c_idx)
                    if new_cand_has_bw >= idle_value:
                        self.record[t_idx, new_s_idx, c_idx] -= idle_value
                        self.t_s_record[t_idx, new_s_idx] -= idle_value
                        self.record[t_idx, s_idx, c_idx] += idle_value
                        self.t_s_record[t_idx, s_idx] += idle_value
                        return True
                    else:
                        self.record[t_idx, new_s_idx, c_idx] -= new_cand_has_bw
                        self.t_s_record[t_idx, new_s_idx] -= new_cand_has_bw
                        self.record[t_idx, s_idx, c_idx] += new_cand_has_bw
                        self.t_s_record[t_idx, s_idx] += new_cand_has_bw
                        idle_value = idle_value - new_cand_has_bw
            for new_s_idx in new_s_idx_list:
                # at t_idx, new_s_idx can be >95%,  remove new_s_idx, then fill it, then find new (recursive)
                if self.is_after_95(new_s_idx, t_idx):
                    if self.fill_one_idle_after_95(new_s_idx, t_idx, idle_value, layer+1):
                        return True
    
    def dispatch_2_one_server(self, input_demand=None):
        if input_demand is not None:
            inputed_client_demand = input_demand
        else:
            inputed_client_demand = client_demand
        for (t_idx, c_idx), demand in self.get_max_idx_gen(inputed_client_demand):
            s_list = self.qos_avail_for_c[c_idx]
            occu_5_num = []
            occu_5_num = [ len(self.server_5_t_idx[s_idx])-(t_idx in self.server_5_t_idx[s_idx]) for s_idx in s_list ]
            # arg = np.argsort(np.array(occu_5_num))
            # s_arr = np.array(s_list)[arg]
            if demand == 0: continue  # don't need it 
            s_arr = self._get_s_idx_arr(s_list, occu_5_num, t_idx)
            for idx, s_idx in enumerate(s_arr):
                if t_idx in self.server_5_t_idx[s_idx]: # in server top 5, put all the resources into
                    if self.server_5_value[s_idx][t_idx] == bandwidth[s_idx]: # server is full at current time, next loop
                        continue
                    else: # server is not full, try to fill it to full
                        left, assign_bandwidth = self.assign(t_idx, s_idx, c_idx, demand)
                        self.server_5_value[s_idx][t_idx] += assign_bandwidth
                        demand = left
                        if left: continue
                        else: break
                elif len(self.server_5_t_idx[s_idx]) != self.higher_95_num: # not in server top 5, top 5 is not full, fill a blank
                    self.server_5_t_idx[s_idx].add(t_idx)
                    left, assign_bandwidth = self.assign(t_idx, s_idx, c_idx, demand)
                    self.server_5_value[s_idx][t_idx] = assign_bandwidth
                    demand = left
                    if left: continue
                    else: break
                else:  # not in top 5, top 5 is full, put average in all the avail
                    avg_s_arr = s_arr[idx:]
                    avg_dispatch = math.floor(demand / len(avg_s_arr))
                    remain = demand - avg_dispatch * len(avg_s_arr)
                    for ss_idx in avg_s_arr:
                        left, _ = self.assign(t_idx, ss_idx, c_idx, avg_dispatch + remain)
                        remain = left
                        if left: continue
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
        # arr = self.record[:, s_idx, :].sum(axis=-1)
        arr = self.t_s_record[:, s_idx].copy()
        # arr = self.record2[:, s_idx, c_idx].copy()
        i = 0
        while i < self.higher_95_num:
            cand = np.argmax(arr)
            if cand == t_idx: 
                return True
            arr[cand] = 0
            i += 1
        return False
    
    def get_batch_after_95(self, arr: np.ndarray):  # time * server
        higher_than_95_idx_list = []
        sep_idx = self.higher_95_num
        for time_series in arr.T:
            idx = np.argpartition(time_series, -sep_idx)[-sep_idx:]
            higher_than_95_idx_list.append(idx)
        return higher_than_95_idx_list
    
    def get_batch_prev_95(self, arr: np.ndarray):
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
        barrier_list = [ 0 for _ in range(s_len) ]
        out_t_idx_list = []
        out_s_idx_list = []
        out_res_at_95 = []
        value_95 = np.array(value_at_95_list)  # 95 value for each server
        idx_for_95 = np.argsort(np.array(value_at_95_list))[::-1]  # index for 95 value (sorting)
        ini_val = value_95[idx_for_95[0]]
        value_94 = np.array(value_at_94_list)
        for i in range(s_len):
            s_idx = idx_for_95[i]
            curr_val = value_95[s_idx]
            if curr_val == 0: break
            if ini_val / curr_val > 20: break
            t_idx = t_idx_95_list[s_idx]
            barrier_list[s_idx] = value_94[s_idx]
            # if (t_idx, s_idx) in self.forbidden:
            #     continue
            out_t_idx_list.append(t_idx)
            out_s_idx_list.append(s_idx)
            out_res_at_95.append(curr_val)
        for j in range(i, s_len):
            s_idx = idx_for_95[i]
            barrier_list[s_idx] = value_95[s_idx]
        return (out_t_idx_list, out_s_idx_list, out_res_at_95), barrier_list
    
    def dispatch_to_small(self, barrier_list: List[int], t_idx: int, s_idx: int):
        barrier = barrier_list[s_idx]
        # sum_at_here = self.record[t_idx, s_idx].sum()  
        # return max(barrier - sum_at_here - 1, 0)
        return max(barrier - self.t_s_record[t_idx, s_idx], 0)  # TODO: why -1?
    
    def analyse_larger_than_95(self):
        self.t_c_larger_95_not_full_rec = {}  # (t_idx, c_idx) -> (s_idx, can_fill_bw)
        for s_idx in range(s_len):
            upper_limit = bandwidth[s_idx]
            for t_idx in self.server_5_t_idx[s_idx]:
                c_series = self.record[t_idx, s_idx]
                c_idx_arr = np.arange(c_len)[c_series > 0]
                for c_idx in c_idx_arr:
                    can_fill_bw = upper_limit - self.t_s_record[t_idx, s_idx]
                    self.t_c_larger_95_not_full_rec[(t_idx, c_idx)] = (s_idx, can_fill_bw)
    
    def try_fill_larger_than_95(self, t_idx: int, from_s_idx: int, c_idx: int, provide: int) -> Tuple[int, int]:  # left, filled
        to_s_idx, can_be_fill = self.t_c_larger_95_not_full_rec.get((t_idx, c_idx), (None, 0))
        if can_be_fill:
            vary_value = min(provide, can_be_fill)
            self.t_s_include_c[t_idx][to_s_idx].add(c_idx)
            self.record[t_idx, to_s_idx, c_idx] += vary_value
            self.t_s_record[t_idx, to_s_idx] += vary_value
            self.record[t_idx, from_s_idx, c_idx] -= vary_value
            self.t_s_record[t_idx, from_s_idx] -= vary_value
            # TODO: to update self.server_5_value
            return provide - vary_value
        return provide
    
    def index_of(self, perc: float) -> int:
        return math.ceil(t_len * perc) - 1

    def _get_95_and_barrier_for_s(self, barrier_perc: float): 
        # arr = self.record.sum(axis=-1).T  # arr: s_idx, t_idx
        arr = self.t_s_record.T
        idx_barrier = self.index_of(barrier_perc)
        idx = np.argpartition(arr, (idx_barrier, self.idx_95))  # s_idx, t_idx
        out_idx = idx[:, idx_barrier+1: self.idx_95 + 1]  # idx for t  # TODO: may have problem
        idx = idx[:, [idx_barrier, self.idx_95]]
        idx_1 = np.tile(np.arange(s_len).reshape(-1, 1), 2)
        values_barrier, values_95 = arr[idx_1, idx].T
        return out_idx, values_barrier, values_95
    
    def dispatch_again_batch_for_one_server(self, barrier_perc=0.8):
        can_cut_t_idxs, values_barrier, values_95 = self._get_95_and_barrier_for_s(barrier_perc)
        prior_idx = np.argsort(values_barrier - values_95)  # s_idx
        barrier_in_progress = values_95.copy()
        added_2_prev95_obj = defaultdict(int)
        for s_idx_orig in prior_idx:
            barrier = values_barrier[s_idx_orig]
            barrier_in_progress[s_idx_orig] = barrier
            res_at_95 = values_95[s_idx_orig]
            # can_move_value = res_at_95 - barrier
            # can_move_perc = can_move_value / res_at_95
            if res_at_95 == 0: continue
            for t_idx in can_cut_t_idxs[s_idx_orig]:   # find t_idx in prec% ~ 95%
                can_move_value = self.t_s_record[t_idx, s_idx_orig] - barrier
                can_move_perc = can_move_value / res_at_95
                for c_idx, res_at_c in enumerate(self.record[t_idx, s_idx_orig]):  # in this t_idx, contains c_idx
                    demand = math.ceil(res_at_c * can_move_perc)
                    for s_idx_new in self.qos_avail_for_c[c_idx]:
                        if demand <= 0: break
                        if s_idx_new == s_idx_orig: continue
                        # demand = self.try_fill_larger_than_95(t_idx, s_idx_orig, c_idx, demand)
                        dispatch_minus = added_2_prev95_obj.get((t_idx, s_idx_new), 0)
                        can_dispatch = self.dispatch_to_small(barrier_in_progress, t_idx, s_idx_new)
                        if can_dispatch > dispatch_minus:
                            assign_bw = min(demand, can_dispatch - dispatch_minus) 
                            demand -= assign_bw
                            self.assign(t_idx, s_idx_new, c_idx, assign_bw)
                            self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                            self.t_s_record[t_idx, s_idx_orig] -= assign_bw
                            added_2_prev95_obj[(t_idx, s_idx_new)] += assign_bw
                    if demand:
                        for s_idx_new in self.s2s_bridge[s_idx_orig]:  # find a new server
                            for c_idx_bridge, can_exchange in enumerate(self.record[t_idx, s_idx_new]):  # use a bridge c_idx
                                if c_idx_bridge not in self.qos_avail_for_s[s_idx_orig]: continue
                                for s_idx_final in self.qos_avail_for_c[c_idx_bridge]:
                                    if c_idx not in self.qos_avail_for_s[s_idx_final]: continue
                                    if demand <= 0: break
                                    if s_idx_final == s_idx_orig: continue
                                    dispatch_minus = added_2_prev95_obj.get((t_idx, s_idx_final), 0)
                                    can_dispatch = self.dispatch_to_small(barrier_in_progress, t_idx, s_idx_final)
                                    if can_dispatch > dispatch_minus:
                                        assign_bw = min(demand, can_dispatch - dispatch_minus, can_exchange) 
                                        demand -= assign_bw
                                        # demand: orig to final
                                        self.assign(t_idx, s_idx_final, c_idx, assign_bw)
                                        self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                                        self.t_s_record[t_idx, s_idx_orig] -= assign_bw
                                        # exchange: new to orig
                                        self.assign(t_idx, s_idx_orig, c_idx_bridge, assign_bw)
                                        self.record[t_idx, s_idx_new, c_idx_bridge] -= assign_bw
                                        self.t_s_record[t_idx, s_idx_new] -= assign_bw
                                        added_2_prev95_obj[(t_idx, s_idx_final)] += assign_bw
    
    def dispatch_again_batch_for_multi_server(self, barrier_perc=0.8):
        can_cut_t_idxs, values_barrier, values_95 = self._get_95_and_barrier_for_s(barrier_perc)
        prior_idx = np.argsort(values_barrier - values_95)  # s_idx
        added_2_prev95_obj = {}
        for s_idx_orig in prior_idx:
            res_at_95 = values_95[s_idx_orig]
            can_move_threshould = res_at_95 * 0.03
            for t_idx in can_cut_t_idxs[s_idx_orig]:
                for c_idx, res in enumerate(self.record[t_idx, s_idx_orig]):
                    if res > can_move_threshould: # ratio of client in this server > 3%
                        demand = math.ceil(self.record[t_idx, s_idx_orig, c_idx] * 0.4)
                        for s_idx_new in self.qos_avail_for_c[c_idx]:
                            if demand <= 0: break
                            if s_idx_new == s_idx_orig: continue
                            # demand = self.try_fill_larger_than_95(t_idx, s_idx_orig, c_idx, demand)
                            dispatch_minus = added_2_prev95_obj.get((t_idx, s_idx_new), 0)
                            can_dispatch = self.dispatch_to_small(values_barrier, t_idx, s_idx_new)
                            if can_dispatch > dispatch_minus:
                                assign_bw = min(demand, can_dispatch - dispatch_minus) 
                                demand -= assign_bw
                                self.assign(t_idx, s_idx_new, c_idx, assign_bw)
                                self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                                self.t_s_record[t_idx, s_idx_orig] -= assign_bw
                                added_2_prev95_obj[(t_idx, s_idx_new)] = can_dispatch
        
    def dispatch_again(self):
        # server_t_series = self.record.sum(axis=-1)
        server_t_series = self.t_s_record
        (t_idx_list, s_idx_list, res_at_95_list), barrier_list = self.get_batch_prev_95(server_t_series) # barrier is for each server
        # higher_95_t_idx_for_server = self.get_batch_after_95(server_t_series)
        added_2_prev95_obj = {}
        for t_idx, s_idx_orig, res_at_95 in zip(t_idx_list, s_idx_list, res_at_95_list):
            client_series = self.record[t_idx, s_idx_orig]
            for c_idx, res in enumerate(client_series):
                if res > np.ceil(res_at_95 * 0.03):
                    # if (t_idx, s_idx_orig, c_idx) in self.forbidden: continue
                    demand = np.ceil(self.record[t_idx, s_idx_orig, c_idx] * 0.4).astype('int32')
                    s_idx_cand_list = self.qos_avail_for_c[c_idx]  # server candidate
                    # # move to max 5%
                    demand = self.try_fill_larger_than_95(t_idx, s_idx_orig, c_idx, demand)

                    # for s_idx_new in s_idx_cand_list:
                    #     if demand <= 0: break
                    #     higher_95_t_list = higher_95_t_idx_for_server[s_idx_new]
                    #     if t_idx in higher_95_t_list:
                    #         left, assign_bw = self.assign(t_idx, s_idx_new, c_idx, demand)
                    #         demand = left
                    #         self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                    #         self.t_s_record[t_idx, s_idx_orig] -= assign_bw
                    # move to prev 95%
                    for s_idx_new in s_idx_cand_list:
                        if demand <= 0: break
                        if (t_idx, s_idx_new) in added_2_prev95_obj:
                            dispatch_minus = added_2_prev95_obj[(t_idx, s_idx_new)]
                        else:
                            dispatch_minus = 0
                        if s_idx_new == s_idx_orig: continue
                        can_dispatch = self.dispatch_to_small(barrier_list, t_idx, s_idx_new)
                        if can_dispatch > dispatch_minus:
                            assign_bw = min(demand, can_dispatch - dispatch_minus) 
                            demand -= assign_bw
                            self.assign(t_idx, s_idx_new, c_idx, assign_bw)
                            self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                            self.t_s_record[t_idx, s_idx_orig] -= assign_bw
                            added_2_prev95_obj[(t_idx, s_idx_new)] = can_dispatch
                    # if demand > 0: self.forbidden.add((t_idx, s_idx_orig, c_idx))

    def pre_dispatch_then_dispatch(self, record: np.ndarray):
        my_demand = client_demand.copy()
        for s_idx, t_idx, idle_value in self.idx_of_max_idle(record):
            c_idx_avail_set = set(self.qos_avail_for_s[s_idx])
            c_idx_used_set = set([ c_idx for c_idx, v in enumerate(record[t_idx, s_idx]) if v ])
            new_c_idx_set = c_idx_avail_set - c_idx_used_set
            for c_idx in new_c_idx_set:
                can_dispatch = min(idle_value, my_demand[t_idx, c_idx])
                if can_dispatch == 0: continue
                left, assigned_bw = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                self.server_5_t_idx[s_idx].add(t_idx)
                self.server_5_value[s_idx][t_idx] += assigned_bw
                idle_value -= assigned_bw
                my_demand[t_idx, c_idx] -= assigned_bw
                if idle_value == 0: break
        self.dispatch_2_one_server(my_demand)

def fill_task(s: Solution):
    s.fill_idle_after_95()

def cut_task(s: Solution):
    s.dispatch_again()


if __name__ == '__main__':
    start_time = time.time()
    get_data()
    s = Solution()
    s.dispatch_2_one_server()
    s.analyse_larger_than_95()
    if LOCAL: 
        s.check_output_valid()
        s.calc_score95(True)
    
    
    # if True:
    #     s2 = Solution()
    #     s2.pre_dispatch_then_dispatch(s.record)
    #     if LOCAL: s2.check_output_valid()
    #     s2.calc_score95(True)
    # s = s2

    if LOCAL: 
        s.check_output_valid()
        time_threshould = 10
    else: 
        time_threshould = 285

    # s.dispatch_again_batch_for_one_server(0.8)
    # s.calc_score95(True)
    
    # print('after batch cut')

    prev_score = s.calc_score95(print_sep=False)
    while time.time() - start_time < time_threshould:
        s.dispatch_again()
        curr_score = s.calc_score95(print_sep=False)
        if (prev_score - curr_score) / curr_score < 0.00000003: 
            break
        prev_score = curr_score
    print('iterate end. \n\n\n')

    # s.dispatch_again_batch_for_one_server(0.85)

    prev_score = s.calc_score95(False)
    # for i in np.arange(0.93, 0.70, -0.02):
    #     s.dispatch_again_batch_for_one_server(i)
    #     print(i, end=':  ')
    #     curr_score = s.calc_score95(print_sep=False)
    #     if (prev_score - curr_score) / curr_score < 0.000003: 
    #         break
    #     prev_score = curr_score

    # print('batch cut:')
    # s.dispatch_again_batch_for_multi_server(0.94)
    # if LOCAL: s.calc_score95(True)
    # print('batch cut finished')



    # task = 'cut'
    # while time.time() - start_time < time_threshould:
    #     if task == 'cut':
    #         s.dispatch_again()
    #     else:
    #         s.fill_idle_after_95()
    #     curr_score = s.calc_score95(print_sep=False)
    #     if curr_score == prev_score:
    #         print('score are the same, break')
    #         break
    #     if (prev_score - curr_score) / curr_score < 0.0003: 
    #         if task == 'cut':
    #             task = 'fill'
    #             print('change to fill task.')
    #         else:
    #             task = 'cut'
    #             print('change to cut task.')
    #     prev_score = curr_score


    s.output()
    if LOCAL: s.check_output_valid()