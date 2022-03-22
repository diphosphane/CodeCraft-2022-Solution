from collections import defaultdict
from typing import List, Tuple, Set
from subprocess import getoutput
import math
import time
from dinic_git import Dinic as Graph
# from EK import Graph
from functools import reduce
from read_data import *
import numpy as np

cname, sname, qos, qos_lim = None, None, None, None
start_time = 0
t_len, s_len, c_len = 0, 0, 0
time_label = None
client_demand = None
bandwidth = None
start_time = None
LOCAL = getoutput('uname') == 'Darwin'

def get_data():
    global cname, sname, qos, qos_lim, bandwidth, client_demand, time_label, t_len, s_len, c_len
    cname, sname, qos = read_qos()
    qos_lim = read_qos_limit(); qos = np.array(qos)
    time_label, client_name, client_demand = read_demand()
    server_name, server_bandwidth = read_server_bandwidth()
    bandwidth = np.array([ server_bandwidth[server_name.index(s)] for s in sname ])
    client_idx_list = [ client_name.index(c) for c in cname ]
    client_demand = np.array(client_demand)[:, np.array(client_idx_list)]
    t_len, s_len, c_len = len(time_label), len(sname), len(cname)

class Solution():
    def __init__(self) -> None:
        self.init_95()
        self.init_qos()
        self.record = np.zeros((t_len, s_len, c_len), dtype=np.int32)
        self.t_s_record = np.zeros((t_len, s_len), dtype=np.int32)
        self.t_s_include_c = [ [ set() for _ in range(s_len) ] for _ in range(t_len) ]
        self.after_95_t_4_s = [ set() for _ in range(s_len) ]
        self.gen_time = 0
        self.assign_time = 0
    
    def init_qos(self):
        def _qos4c(c_idx: int) -> List[int]:
            c_qos = qos[:, c_idx]
            qos_avail = c_qos < qos_lim
            out = [ s_idx for s_idx, avail in enumerate(qos_avail) if avail ]
            return set(out)
        
        def _qos4s(s_idx: int) -> List[int]:
            s_qos = qos[s_idx, :]
            qos_avail = s_qos < qos_lim
            out = [ c_idx for c_idx, avail in enumerate(qos_avail) if avail ]
            return set(out)

        self.qos_avail_for_c = [ _qos4c(c_idx) for c_idx in range(c_len) ]
        self.qos_avail_for_s = [ _qos4s(s_idx) for s_idx in range(s_len) ]
        self.avail_s_count = 0
        for each in self.qos_avail_for_s:
            if each: self.avail_s_count += 1
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

    def output(self, record=None):
        if not record:
            record = self.record
        if LOCAL: self.f = open('output/solution.txt', 'w')
        else: self.f = open('/output/solution.txt', 'w')
        for each_time_step_operation in record:
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
        bw_each_time = self.t_s_record.copy()
        bw_each_time.sort(axis=0)
        score_95 = bw_each_time[self.idx_95, :]
        after_95 = bw_each_time[self.idx_95+1:, :].sum(0)
        after_95_sum = after_95.sum()
        final_score = score_95.sum()
        if print_sep:
            print(f'95% score sum: {final_score}\n{sorted(score_95, reverse=True)}\n')
            print(f'after 95 sum: {after_95_sum}\n{sorted(after_95, reverse=True)}')
        else:
            print(f'95% score sum: {final_score}')
            print(f'after 95 sum: {after_95_sum}')
        return final_score
    
    def get_max_idx_gen(self, array: np.ndarray) -> Tuple[Tuple[int, int], int]:
        arr = array.copy()
        cnt = 0; whole_num = reduce(lambda x,y: x*y, arr.shape)
        while cnt < whole_num:
            st = time.time()
            idx = np.unravel_index(np.argmax(arr), arr.shape)
            self.gen_time += (time.time() - st)
            yield idx, arr[idx]
            arr[idx] = 0
            cnt += 1
    
    @staticmethod
    def max_idx_of(arr: np.ndarray) -> Tuple[int, int]:
        return np.unravel_index(np.argmax(arr), arr.shape)
    
    def assign(self, t_idx: int, s_idx: int, c_idx: int, demand: int) -> Tuple[int, int]: # left, assigned_value
        add_up = self.t_s_record[t_idx, s_idx] + demand
        upper_limit = bandwidth[s_idx]
        if add_up > upper_limit: # assign fail
            left = add_up - upper_limit
            assign_bandwidth = demand - left
            if assign_bandwidth != 0: self.t_s_include_c[t_idx][s_idx].add(c_idx)
            self.record[t_idx, s_idx, c_idx] += assign_bandwidth
            st = time.time()
            self.t_s_record[t_idx, s_idx] += assign_bandwidth
            self.assign_time += (time.time() - st)
            return left, assign_bandwidth
        self.record[t_idx, s_idx, c_idx] += demand
        st = time.time()
        self.t_s_record[t_idx, s_idx] += demand
        self.assign_time += (time.time() - st)
        if demand != 0: self.t_s_include_c[t_idx][s_idx].add(c_idx)
        return 0, demand
    
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
            if res_at_95 == 0: continue
            for t_idx in can_cut_t_idxs[s_idx_orig]:   # find t_idx in prec% ~ 95%
                can_move_value = self.t_s_record[t_idx, s_idx_orig] - barrier
                can_move_perc = can_move_value / res_at_95
                for c_idx, res_at_c in enumerate(self.record[t_idx, s_idx_orig]):  # in this t_idx, contains c_idx
                    demand = math.ceil(res_at_c * can_move_perc)
                    for s_idx_new in self.qos_avail_for_c[c_idx]:
                        if demand <= 0: break
                        if s_idx_new == s_idx_orig: continue
                        dispatch_minus = added_2_prev95_obj.get((t_idx, s_idx_new), 0)
                        can_dispatch = self.dispatch_to_small(barrier_in_progress, t_idx, s_idx_new)
                        if can_dispatch > dispatch_minus:
                            assign_bw = min(demand, can_dispatch - dispatch_minus) 
                            demand -= assign_bw
                            self.assign(t_idx, s_idx_new, c_idx, assign_bw)
                            self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                            self.t_s_record[t_idx, s_idx_orig] -= assign_bw
                            added_2_prev95_obj[(t_idx, s_idx_new)] += assign_bw
                    # 2nd jump assign
                    # if demand:
                    #     for s_idx_new in self.s2s_bridge[s_idx_orig]:  # find a new server
                    #         for c_idx_bridge, can_exchange in enumerate(self.record[t_idx, s_idx_new]):  # use a bridge c_idx
                    #             if c_idx_bridge not in self.qos_avail_for_s[s_idx_orig]: continue
                    #             for s_idx_final in self.qos_avail_for_c[c_idx_bridge]:
                    #                 if c_idx not in self.qos_avail_for_s[s_idx_final]: continue
                    #                 if demand <= 0: break
                    #                 if s_idx_final == s_idx_orig: continue
                    #                 dispatch_minus = added_2_prev95_obj.get((t_idx, s_idx_final), 0)
                    #                 can_dispatch = self.dispatch_to_small(barrier_in_progress, t_idx, s_idx_final)
                    #                 if can_dispatch > dispatch_minus:
                    #                     assign_bw = min(demand, can_dispatch - dispatch_minus, can_exchange) 
                    #                     demand -= assign_bw
                    #                     # demand: orig to final
                    #                     self.assign(t_idx, s_idx_final, c_idx, assign_bw)
                    #                     self.record[t_idx, s_idx_orig, c_idx] -= assign_bw
                    #                     self.t_s_record[t_idx, s_idx_orig] -= assign_bw
                    #                     # exchange: new to orig
                    #                     self.assign(t_idx, s_idx_orig, c_idx_bridge, assign_bw)
                    #                     self.record[t_idx, s_idx_new, c_idx_bridge] -= assign_bw
                    #                     self.t_s_record[t_idx, s_idx_new] -= assign_bw
                    #                     added_2_prev95_obj[(t_idx, s_idx_final)] += assign_bw
    
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

    def dispatch_again(self):
        server_t_series = self.t_s_record
        (t_idx_list, s_idx_list, res_at_95_list), barrier_list = self.get_batch_prev_95(server_t_series) # barrier is for each server
        added_2_prev95_obj = {}
        for t_idx, s_idx_orig, res_at_95 in zip(t_idx_list, s_idx_list, res_at_95_list):
            client_series = self.record[t_idx, s_idx_orig]
            for c_idx, res in enumerate(client_series):
                if res > np.ceil(res_at_95 * 0.03):
                    demand = np.ceil(self.record[t_idx, s_idx_orig, c_idx] * 0.4).astype('int32')
                    s_idx_cand_list = self.qos_avail_for_c[c_idx]  # server candidate
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
    
    def _restore_idx(self, idx, arr):
        prev_sum = 0; new_idx = idx
        new_sum = arr[:idx+1].sum()
        while prev_sum != new_sum:
            prev_sum = new_sum
            new_idx = idx + new_sum
            new_sum = arr[:new_idx+1].sum()
        return new_idx

    def dispatch_from_server_no_avg(self):
        # DEL:
        self.sum_at_5 = 0
        s_full_filled = np.zeros(s_len, dtype=np.int32)
        after_95_t_includ_s = defaultdict(set)
        self.demand_after_5_dispatch = client_demand.copy()
        qos_bool_c_s_orig = np.array((qos < qos_lim).T, order='F')
        qos_bool_c_s = qos_bool_c_s_orig
        s_idx_resotre_arr = np.zeros(s_len, dtype=np.int32)
        s_idx_deleted = []
        arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s_orig  # t * c  dot  c * s  -->  t * s
        cnt = 0
        st = time.time()
        while cnt < self.higher_95_num * self.avail_s_count:
            t_idx, s_idx = self.max_idx_of(arr_t_s)
            if arr_t_s[t_idx, s_idx] == 0: break
            s_idx = self._restore_idx(s_idx, s_idx_resotre_arr)
            if s_full_filled[s_idx] == self.higher_95_num: 
                s_idx_resotre_arr[s_idx] = 1
                s_idx_deleted.append(s_idx)
                qos_bool_c_s = np.delete(qos_bool_c_s_orig, s_idx_deleted, axis=1)
                arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s  # t * c  dot  c * s  -->  t * s
                continue
            c_avail_set = self.qos_avail_for_s[s_idx]
            if not c_avail_set: continue
            after_95_t_includ_s[t_idx].add(s_idx)
            self.after_95_t_4_s[s_idx].add(t_idx)
            added = 0; left = 0
            for c_idx in self.qos_avail_for_s[s_idx]: # TODO: select c_idx scheme
                if added == bandwidth[s_idx]: break
                left, assigned = self.assign(t_idx, s_idx, c_idx, self.demand_after_5_dispatch[t_idx, c_idx])
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                self.sum_at_5 += assigned  # DEL
                added += assigned
            arr_t_s[t_idx] = self.demand_after_5_dispatch[t_idx].dot(qos_bool_c_s)
            s_full_filled[s_idx] += 1
            cnt += 1
        print(f'matrix used time: {time.time() - st}')
        self.after_95_record = self.record.copy()
        st = time.time()
        for (t_idx, c_idx), need_dispatch in self.get_max_idx_gen(self.demand_after_5_dispatch):
            s_avail_set = self.qos_avail_for_c[c_idx]
            s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
            remain = need_dispatch
            for s_idx in s_avail_set:
                remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
                if remain == 0: break
            if remain: 
                raise BaseException('not fully dispatched')
        print(f'remain used time: {time.time() - st}')

    def dispatch_from_server_5_times_no_avg(self):
        # DEL:
        self.sum_at_5 = 0
        s_full_filled = np.zeros(s_len, dtype=np.int32)
        after_95_t_includ_s = defaultdict(set)
        self.demand_after_5_dispatch = client_demand.copy()
        qos_bool_c_s_orig = np.array((qos < qos_lim).T, order='F')
        qos_bool_c_s = qos_bool_c_s_orig
        s_idx_resotre_arr = np.zeros(s_len, dtype=np.int32)
        s_idx_deleted = []
        arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s_orig  # t * c  dot  c * s  -->  t * s
        cnt = 0
        st = time.time()
        while cnt < self.higher_95_num * self.avail_s_count:
            t_idx, s_idx = self.max_idx_of(arr_t_s)
            if arr_t_s[t_idx, s_idx] == 0: break
            t_idx_arr = np.argpartition(arr_t_s[:, s_idx], -5)[-5:]
            s_idx = self._restore_idx(s_idx, s_idx_resotre_arr)
            if s_full_filled[s_idx] == self.higher_95_num: 
                s_idx_resotre_arr[s_idx] = 1
                s_idx_deleted.append(s_idx)
                qos_bool_c_s = np.delete(qos_bool_c_s_orig, s_idx_deleted, axis=1)
                arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s  # t * c  dot  c * s  -->  t * s
                continue
            for t_idx in t_idx_arr:
                c_avail_set = self.qos_avail_for_s[s_idx]
                if not c_avail_set: continue
                after_95_t_includ_s[t_idx].add(s_idx)
                self.after_95_t_4_s[s_idx].add(t_idx)
                added = 0; left = 0
                for c_idx in self.qos_avail_for_s[s_idx]: # TODO: select c_idx scheme
                    if added == bandwidth[s_idx]: break
                    left, assigned = self.assign(t_idx, s_idx, c_idx, self.demand_after_5_dispatch[t_idx, c_idx])
                    self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                    self.sum_at_5 += assigned  # DEL
                    added += assigned
                arr_t_s[t_idx] = self.demand_after_5_dispatch[t_idx].dot(qos_bool_c_s)
                s_full_filled[s_idx] += 1
                cnt += 1
        print(f'matrix used time: {time.time() - st}')
        self.after_95_record = self.record.copy()
        st = time.time()
        for (t_idx, c_idx), need_dispatch in self.get_max_idx_gen(self.demand_after_5_dispatch):
            s_avail_set = self.qos_avail_for_c[c_idx]
            s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
            remain = need_dispatch
            for s_idx in s_avail_set:
                remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
                if remain == 0: break
            if remain: 
                raise BaseException('not fully dispatched')
        print(f'remain used time: {time.time() - st}')

    def dispatch_from_server_5_times(self):
        # DEL:
        self.sum_at_5 = 0
        s_full_filled = np.zeros(s_len, dtype=np.int32)
        after_95_t_includ_s = defaultdict(set)
        self.demand_after_5_dispatch = client_demand.copy()
        qos_bool_c_s_orig = np.array((qos < qos_lim).T, order='F')
        qos_bool_c_s = qos_bool_c_s_orig
        s_idx_resotre_arr = np.zeros(s_len, dtype=np.int32)
        s_idx_deleted = []
        arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s_orig  # t * c  dot  c * s  -->  t * s
        cnt = 0
        st = time.time()
        while cnt < self.higher_95_num * self.avail_s_count:
            t_idx, s_idx = self.max_idx_of(arr_t_s)
            if arr_t_s[t_idx, s_idx] == 0: break
            t_idx_arr = np.argpartition(arr_t_s[:, s_idx], -5)[-5:]
            s_idx = self._restore_idx(s_idx, s_idx_resotre_arr)
            if s_full_filled[s_idx] == self.higher_95_num: 
                s_idx_resotre_arr[s_idx] = 1
                s_idx_deleted.append(s_idx)
                qos_bool_c_s = np.delete(qos_bool_c_s_orig, s_idx_deleted, axis=1)
                arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s  # t * c  dot  c * s  -->  t * s
                continue
            for t_idx in t_idx_arr:
                c_avail_set = self.qos_avail_for_s[s_idx]
                if not c_avail_set: continue
                after_95_t_includ_s[t_idx].add(s_idx)
                self.after_95_t_4_s[s_idx].add(t_idx)
                added = 0; left = 0
                for c_idx in self.qos_avail_for_s[s_idx]: # TODO: select c_idx scheme
                    if added == bandwidth[s_idx]: break
                    left, assigned = self.assign(t_idx, s_idx, c_idx, self.demand_after_5_dispatch[t_idx, c_idx])
                    self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                    self.sum_at_5 += assigned  # DEL
                    added += assigned
                arr_t_s[t_idx] = self.demand_after_5_dispatch[t_idx].dot(qos_bool_c_s)
                s_full_filled[s_idx] += 1
                cnt += 1
        print(f'matrix used time: {time.time() - st}')
        self.after_95_record = self.record.copy()
        st = time.time()
        for (t_idx, c_idx), need_dispatch in self.get_max_idx_gen(self.demand_after_5_dispatch):
            s_avail_set = self.qos_avail_for_c[c_idx]
            s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
            avg = need_dispatch // len(s_avail_set)
            remain = need_dispatch - avg * len(s_avail_set)
            for s_idx in s_avail_set:
                remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
            if remain: 
                for s_idx in s_avail_set:
                    remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
            if remain: 
                raise BaseException('not fully dispatched')
        print(f'remain used time: {time.time() - st}')

    def dispatch_from_server(self):
        # DEL:
        self.sum_at_5 = 0
        s_full_filled = np.zeros(s_len, dtype=np.int32)
        after_95_t_includ_s = defaultdict(set)
        self.demand_after_5_dispatch = client_demand.copy()
        qos_bool_c_s_orig = np.array((qos < qos_lim).T, order='F')
        qos_bool_c_s = qos_bool_c_s_orig
        s_idx_resotre_arr = np.zeros(s_len, dtype=np.int32)
        s_idx_deleted = []
        arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s_orig  # t * c  dot  c * s  -->  t * s
        cnt = 0
        st = time.time()
        while cnt < self.higher_95_num * self.avail_s_count:
            t_idx, s_idx = self.max_idx_of(arr_t_s)
            if arr_t_s[t_idx, s_idx] == 0: break
            s_idx = self._restore_idx(s_idx, s_idx_resotre_arr)
            if s_full_filled[s_idx] == self.higher_95_num: 
                s_idx_resotre_arr[s_idx] = 1
                s_idx_deleted.append(s_idx)
                qos_bool_c_s = np.delete(qos_bool_c_s_orig, s_idx_deleted, axis=1)
                arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s  # t * c  dot  c * s  -->  t * s
                continue
            c_avail_set = self.qos_avail_for_s[s_idx]
            if not c_avail_set: continue
            after_95_t_includ_s[t_idx].add(s_idx)
            self.after_95_t_4_s[s_idx].add(t_idx)
            added = 0; left = 0
            for c_idx in self.qos_avail_for_s[s_idx]: # TODO: select c_idx scheme
                if added == bandwidth[s_idx]: break
                left, assigned = self.assign(t_idx, s_idx, c_idx, self.demand_after_5_dispatch[t_idx, c_idx])
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                self.sum_at_5 += assigned  # DEL
                added += assigned
            arr_t_s[t_idx] = self.demand_after_5_dispatch[t_idx].dot(qos_bool_c_s)
            s_full_filled[s_idx] += 1
            cnt += 1
        print(f'matrix used time: {time.time() - st}')
        self.after_95_record = self.record.copy()
        st = time.time()
        for (t_idx, c_idx), need_dispatch in self.get_max_idx_gen(self.demand_after_5_dispatch):
            s_avail_set = self.qos_avail_for_c[c_idx]
            s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
            avg = need_dispatch // len(s_avail_set)
            remain = need_dispatch - avg * len(s_avail_set)
            for s_idx in s_avail_set:
                remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
            if remain: 
                for s_idx in s_avail_set:
                    remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
            if remain: 
                raise BaseException('not fully dispatched')
        print(f'remain used time: {time.time() - st}')

    def _max_idx_4_row(self, arr: np.ndarray):
        idx = np.argmax(arr, axis=1)
        return [ self._restore_idx(sel_idx, self.removed_t_idx_4_s[i]) for i, sel_idx in enumerate(idx)]
    
    def _del_col_4_arr(self, arr, del_arr):
        row_cnt = arr.shape[0]
        del_arr_here = del_arr.reshape(row_cnt, -1)
        col_cnt = arr.shape[1] - del_arr_here.reshape(row_cnt, -1).shape[1]
        out_arr = np.empty((row_cnt, col_cnt), dtype=np.int32)
        for r_idx in range(row_cnt):
            real_idx = [ self._restore_idx(sel_idx, self.removed_t_idx_4_s[i]) for i, sel_idx in enumerate(del_arr_here)]
            out_arr[r_idx] = np.delete(arr[r_idx], real_idx)
            pass


    def dispatch_again_3_block(self):
        # self.removed_t_idx_4_s = np.zeros((s_len, t_len), dtype=np.int32)
        self.removed_t_s = np.zeros((t_len, s_len), dtype=np.int32)
        for s_idx, t_set in enumerate(self.after_95_t_4_s):
            for t_idx in t_set:
                self.removed_t_s[t_idx, s_idx] = 1
                # self.removed_t_idx_4_s[s_idx, t_idx] = 1
        max_t_idx_active_4_s = np.argmax(np.ma.array(self.t_s_record, self.removed_t_s), axis=0)  # t for each s
        max_value = self.t_s_record[max_t_idx_active_4_s, np.arange(s_len)]
        tmp_removed_t_s = self.removed_t_s.copy()
        for s_idx, t_idx in enumerate(max_t_idx_active_4_s):
            tmp_removed_t_s[t_idx, s_idx] = 1
        barrier_idx = np.argmax(np.ma.array(self.t_s_record, tmp_removed_t_s), axis=0)
        barrier_value = self.t_s_record[barrier_idx, np.arange(s_len)]
        diff = max_value - barrier_value
        for s_idx, diff_value in enumerate(diff):
            added_2_prev95_obj = defaultdict(int)  # t,s -> added_value
            active_t = barrier_idx[s_idx]
            

            local_record = []  # t, s
            local_added_2_prev_95_obj = defaultdict() # 
            pass


        active_t_for_server = self._max_idx_4_row()
        removed_t_for_server = self.after_95_t_4_s
        active_t_for_server = []
        to_be_added_t_for_server = []
        for s_idx in range(s_len):
            
            pass


    def dispatch_again_3_block(self):
        self.removed_t_idx_4_s = [ [] for _ in range(s_len) ]
        for s_idx, t_set in enumerate(self.after_95_t_4_s):
            for t_idx in t_set:
                self.removed_t_idx_4_s[s_idx].append(t_idx)
        active_t_4_s = [ [] for _ in range(s_len) ]
        
    def convert2graph_old(self):
        self.graph = Graph()
        # define vertex
        for t in range(t_len+1):  # idx in t_len is gather node for s (used to limit bandwidth)
            self.graph.add([ f'{t}s{s}' for s in range(s_len) ])
            self.graph.add([ f'{t}c{c}' for c in range(c_len) ])
            self.graph.add([ f'{t}e{c}' for c in range(c_len) ])
        self.graph.add(['s', 't'])
        self.values_95_for_s = np.argpartition(self.t_s_record, self.idx_95, axis=0)[self.idx_95]
        for t in range(t_len):
            for s in range(s_len):
                self.graph.add_edge('s', f'{t_len}s{s}', self.values_95_for_s[s])
                self.graph.add_edge(f'{t_len}s{s}', f'{t}s{s}', self.values_95_for_s[s])
            for c in range(c_len):
                self.graph.add_edge(f'{t}c{c}', f'{t}e{c}', self.demand_after_5_dispatch[t, c]) # e means end
                self.graph.add_edge(f'{t}e{c}', 't', client_demand[t, c])
        for s_idx, t_set in enumerate(self.after_95_t_4_s):
            for t_idx in t_set:
                self.graph.add_edge(f'{t_len}s{s_idx}', f'{t_idx}s{s_idx}', 0)
        for t in range(t_len):
            for s in range(s_len):
                c_list = self.qos_avail_for_s[s]
                for c in c_list:
                    self.graph.add_edge(f'{t}s{s}', f'{t}c{c}', min(self.demand_after_5_dispatch[t, c], bandwidth[s]))

    def _get_demand_out(self, g: Graph):
        c_demand = np.zeros(c_len, dtype=np.int32)
        for c in range(c_len):
            c_demand[c] = g.get_flow(f'c{c}', 't')
        return c_demand
    
    def iterate_s_cap(self):
        print(f'before iterate s: {self.values_95_for_s.sum()} \n', sorted(self.values_95_for_s, reverse=True))
        arg = np.argsort(self.values_95_for_s)
        not_reduced_list = []
        for s_idx in arg:
            value = self.values_95_for_s[s_idx]
            # if cnt == 20: break
            # cnt += 1
            if value == 0: continue
            test_value = value
        # for s_idx, value in enumerate(self.values_95_for_s):
            flow_diff_max = 0
            test_value = test_value // 2
            for t_idx, g in enumerate(self.graph4t):
                g.add_edge('s', f's{s_idx}', test_value)
                g.calc_max_flow('s', 't')
                d_out = self._get_demand_out(g)
                flow_diff = self.demand_after_5_dispatch[t_idx].sum() - d_out.sum()
                flow_diff_max = max(flow_diff_max, flow_diff)
            # re-define flow diff
            # flow_sum = np.array([ g.max_flow for g in self.graph4t ]).sum()
            # diff = self.flow_sum - flow_sum
            reduced = value - test_value - flow_diff_max
            if flow_diff_max == 0:
                not_reduced_list.append(s_idx)
            print(f'server index: {s_idx} \t prev: {value} \t test value: {test_value} \t flow diff: {flow_diff_max} \t reduced: {reduced}')
            # if test_value == 0: break
            # if diff:
                # self.values_95_for_s[s_idx] = test_value + diff
                # for g in self.graph4t:
                #     g.add_edge('s', f's{s_idx}', test_value + diff)
            if flow_diff_max:
                self.values_95_for_s[s_idx] = test_value + flow_diff_max
                for g in self.graph4t:
                    g.add_edge('s', f's{s_idx}', test_value + flow_diff_max)
        # second time reduce in network flow
        print('start 2nd time network flow reduce')
        print('iterate list: ', not_reduced_list)
        for s_idx in not_reduced_list:
            flow_diff_max = 0
            test_value = 0
            for t_idx, g in enumerate(self.graph4t):
                g.add_edge('s', f's{s_idx}', test_value)
                g.calc_max_flow('s', 't')
                d_out = self._get_demand_out(g)
                flow_diff = self.demand_after_5_dispatch[t_idx].sum() - d_out.sum()
                flow_diff_max = max(flow_diff_max, flow_diff)
            # re-define flow diff
            # flow_sum = np.array([ g.max_flow for g in self.graph4t ]).sum()
            # diff = self.flow_sum - flow_sum
            reduced = value - test_value - flow_diff_max
            if flow_diff_max == 0:
                not_reduced_list.append(s_idx)
            print(f'server index: {s_idx} prev: {value} \t test value: {test_value} \t flow diff: {flow_diff_max} \t reduced: {reduced}')
            # if test_value == 0: break
            # if diff:
                # self.values_95_for_s[s_idx] = test_value + diff
                # for g in self.graph4t:
                #     g.add_edge('s', f's{s_idx}', test_value + diff)
            if flow_diff_max:
                self.values_95_for_s[s_idx] = test_value + flow_diff_max
                for g in self.graph4t:
                    g.add_edge('s', f's{s_idx}', test_value + flow_diff_max)
            
            pass
        print(f'after iterate s: {self.values_95_for_s.sum()} \n', sorted(self.values_95_for_s, reverse=True))
        for g in self.graph4t:
            g.calc_max_flow('s', 't')
        flow_sum = np.array(self.max_flow_list).sum()
        print(f'final flow sum: {flow_sum}')
        
    def read_out_network(self):
        self.record = self.after_95_record
        for t_idx, g in enumerate(self.graph4t):
            for s_idx in range(s_len):
                for c_idx in range(c_len):
                    v = g.get_flow(f's{s_idx}', f'c{c_idx}')
                    self.record[t_idx, s_idx, c_idx] += v
        self.t_s_record = self.record.sum(axis=-1)

    def construct_all_graph(self):
        t_idx_95 = np.argpartition(self.t_s_record, self.idx_95, axis=0)[self.idx_95]
        self.values_95_for_s = self.t_s_record[t_idx_95, np.arange(s_len)]
        # max_idx = self.higher_95_num + 1
        # self.values_95_for_s = np.zeros(s_len, dtype=np.int32)
        # for s_idx, t_series in enumerate(self.t_s_record.T):
        #     # idx = np.argpartition(t_series, -max_idx)[-max_idx]
        #     t_series.sort()
        #     self.values_95_for_s[s_idx] = t_series[self.idx_95]
        #     # self.values_95_for_s[s_idx] = t_series[idx]
        print(f'95% sum at graph: {np.sum(self.values_95_for_s)}')
        # self.values_95_for_s = self.t_s_record[idxs, np.arange(s_len)]
        self.graph4t: List[Graph] = [ self.construct_graph(t) for t in range(t_len) ]
        for t in range(t_len):
            # self.graph4t[t].max_capacity_augment('s', 't')
            self.graph4t[t].calc_max_flow('s', 't')
        self.max_flow_list = [ g.max_flow for g in self.graph4t ]
        self.flow_sum = np.array(self.max_flow_list).sum()
        print(f'flow sum: {np.array(self.max_flow_list).sum()}')
        print(f'separate flow: \n {sorted(self.max_flow_list, reverse=True)}')
        self.after_95_t_4_s
    
    def construct_graph(self, t) -> Graph:  # when t = t_len is gather point
        graph = Graph()
        graph.add([ f'b{s}' for s in range(s_len) ])  # begin
        graph.add([ f's{s}' for s in range(s_len) ])  # server
        graph.add([ f'c{c}' for c in range(c_len) ])  # client
        graph.add([ f'e{c}' for c in range(c_len) ])  # end
        graph.add(['s', 't'])
        for s in range(s_len):
            graph.add_edge('s', f's{s}', self.values_95_for_s[s])
        for s, t_set in enumerate(self.after_95_t_4_s):
            # if t in t_set: graph.add_edge(f's', f's{s}', bandwidth[s])
            # if t in t_set: graph.add_edge(f's', f's{s}', bandwidth[s]-self.t_s_record[t, s])
            if t in t_set: graph.add_edge(f's', f's{s}', 0)
        for s in range(s_len):
            for c in self.qos_avail_for_s[s]:
                graph.add_edge(f's{s}', f'c{c}', min(self.demand_after_5_dispatch[t, c], bandwidth[s]))
        for c in range(c_len):
            # graph.add_edge(f'c{c}', f't', client_demand[t, c])
            graph.add_edge(f'c{c}', f't', self.demand_after_5_dispatch[t, c])
        return graph

    def max_flow_except_5(self):
        return self.graph.max_flow


if __name__ == '__main__':
    get_data()
    start_time = time.time()
    s = Solution()
    # s.dispatch_from_server_no_avg()
    # s.dispatch_from_server()
    # s.dispatch_from_server_5_times_no_avg()
    s.dispatch_from_server_5_times()
    if LOCAL: 
        print(f'used time normal: {(time.time()-start_time):.2f}')
        s.calc_score95(False)
    print(f'gen time: {s.gen_time}')
    print(f'assign time: {s.assign_time}')

    # s.construct_all_graph()
    # s.iterate_s_cap()
    # s.read_out_network()
        
    print(f'demand except 5% is {s.demand_after_5_dispatch.ravel().sum()}')
    print(f'sum at 5: {s.sum_at_5}, demand - sum_at_5: {client_demand.ravel().sum()-s.sum_at_5}')
    # s.iterate_s_cap()
    
    # s.analyse_larger_than_95()

    if LOCAL: 
        s.check_output_valid()
        s.calc_score95(True)
        time_threshould = 10
    else: 
        time_threshould = 290

    prev_score = s.calc_score95(print_sep=False)
    while time.time() - start_time < time_threshould:
        # s.dispatch_again()
        # s.dispatch_again_batch_for_one_server(i)
        s.dispatch_again()
        curr_score = s.calc_score95(print_sep=False)
        if (prev_score - curr_score) / curr_score < 0.000000001: 
            break
        prev_score = curr_score

    s.output()
    if LOCAL: 
        print(f'used time: {(time.time()-start_time):.2f}')
        s.check_output_valid()