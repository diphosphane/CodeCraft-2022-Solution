from collections import defaultdict
from typing import List, Tuple, Set
from subprocess import getoutput
from itertools import product
from random import shuffle
from copy import deepcopy
import math
import time
# from dinic import Dinic as Graph
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
    
    def init_qos(self):
        def _qos4c(c_idx: int) -> List[int]:
            c_qos = qos[:, c_idx]
            qos_avail = c_qos < qos_lim
            out = [ s_idx for s_idx, avail in enumerate(qos_avail) if avail ]
            return out
        
        def _qos4s(s_idx: int) -> List[int]:
            s_qos = qos[s_idx, :]
            qos_avail = s_qos < qos_lim
            out = [ c_idx for c_idx, avail in enumerate(qos_avail) if avail ]
            return out

        self.qos_avail_for_c = [ _qos4c(c_idx) for c_idx in range(c_len) ]
        self.qos_avail_num_for_c = np.array([ len(i) for i in self.qos_avail_for_c])
        self.qos_avail_for_s = [ _qos4s(s_idx) for s_idx in range(s_len) ]
        self.qos_avail_num_for_s = np.array([ len(i) for i in self.qos_avail_for_s])

        # new_qos_c = []
        # for c_idx in range(c_len):
        #     s_avail = self.qos_avail_for_c[c_idx]
        #     count = np.array([ self.qos_avail_num_for_s[s_idx] for s_idx in s_avail ])
        #     arg = np.argsort(-count)
        #     new_order = [ s_avail[i] for i in arg ]
        #     new_qos_c.append(new_order)
        # self.qos_avail_for_c = new_qos_c

        # new_qos_s = []
        # for s_idx in range(s_len):
        #     c_avail = self.qos_avail_for_s[s_idx]
        #     count = np.array([ self.qos_avail_num_for_c[c_idx] for c_idx in c_avail ])
        #     arg = np.argsort(-count)
        #     new_order = [ c_avail[i] for i in arg ]
        #     new_qos_s.append(new_order)
        # self.qos_avail_for_s = new_qos_s

        self.qos_avail_for_c_set = [ set(s_list) for s_list in self.qos_avail_for_c ]
        self.qos_avail_for_s_set = [ set(c_list) for c_list in self.qos_avail_for_s ]
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
            if np.any(c_demand_at_t - sum_at_each_time): # if c_demand_at_t != sum_at_each_time:
                print(f'client demand is not equal at time {t_idx}')
                print(f'calculated: \n{sum_at_each_time} \n\n required: \n{c_demand_at_t}')
                print(f'difference (calculated_demand - required_demand): \n {sum_at_each_time - c_demand_at_t}')
                exit(1)
        if np.any(demand_sum - client_demand): # if demand_sum != client_demand:
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
    
    @staticmethod
    def max_idx_gen(array: np.ndarray) -> Tuple[Tuple[int, int], int]:
        arr = array.copy()
        cnt = 0; whole_num = reduce(lambda x,y: x*y, arr.shape)
        while cnt < whole_num:
            idx = np.unravel_index(np.argmax(arr), arr.shape)
            value = arr[idx]
            if value == 0: return
            yield idx, value
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
            assigned = demand - left
            if assigned != 0: self.t_s_include_c[t_idx][s_idx].add(c_idx)
            self.record[t_idx, s_idx, c_idx] += assigned
            self.t_s_record[t_idx, s_idx] += assigned
            return left, assigned
        self.record[t_idx, s_idx, c_idx] += demand
        self.t_s_record[t_idx, s_idx] += demand
        if demand != 0: self.t_s_include_c[t_idx][s_idx].add(c_idx)
        return 0, demand
    
    def index_of(self, perc: float) -> int:
        return math.ceil(t_len * perc) - 1

    def _restore_idx(self, idx, arr):
        prev_sum = 0; new_idx = idx
        new_sum = arr[:idx+1].sum()
        while prev_sum != new_sum:
            prev_sum = new_sum
            new_idx = idx + new_sum
            new_sum = arr[:new_idx+1].sum()
        return new_idx

    # def _get_demand_out(self, g: Graph):
    #     c_demand = np.zeros(c_len, dtype=np.int32)
    #     for c in range(c_len):
    #         c_demand[c] = g.get_flow(f'c{c}', 't')
    #     return c_demand
    
    # def iterate_s_cap(self):
    #     print(f'before iterate s: {self.values_95_for_s.sum()} \n', sorted(self.values_95_for_s, reverse=True))
    #     arg = np.argsort(self.values_95_for_s)
    #     not_reduced_list = []
    #     for s_idx in arg:
    #         value = self.values_95_for_s[s_idx]
    #         if value == 0: continue
    #         test_value = value
    #         flow_diff_max = 0
    #         test_value = test_value // 2
    #         for t_idx, g in enumerate(self.graph4t):
    #             g.add_edge('s', f's{s_idx}', test_value)
    #             g.calc_max_flow('s', 't')
    #             d_out = self._get_demand_out(g)
    #             flow_diff = self.demand_after_5_dispatch[t_idx].sum() - d_out.sum()
    #             flow_diff_max = max(flow_diff_max, flow_diff)
    #         # re-define flow diff
    #         reduced = value - test_value - flow_diff_max
    #         if flow_diff_max == 0:
    #             not_reduced_list.append(s_idx)
    #         print(f'server index: {s_idx} \t prev: {value} \t test value: {test_value} \t flow diff: {flow_diff_max} \t reduced: {reduced}')
    #         if flow_diff_max:
    #             self.values_95_for_s[s_idx] = test_value + flow_diff_max
    #             for g in self.graph4t:
    #                 g.add_edge('s', f's{s_idx}', test_value + flow_diff_max)
    #     # second time reduce in network flow
    #     print('start 2nd time network flow reduce')
    #     print('iterate list: ', not_reduced_list)
    #     for s_idx in not_reduced_list:
    #         flow_diff_max = 0
    #         test_value = 0
    #         for t_idx, g in enumerate(self.graph4t):
    #             g.add_edge('s', f's{s_idx}', test_value)
    #             g.calc_max_flow('s', 't')
    #             d_out = self._get_demand_out(g)
    #             flow_diff = self.demand_after_5_dispatch[t_idx].sum() - d_out.sum()
    #             flow_diff_max = max(flow_diff_max, flow_diff)
    #         # re-define flow diff
    #         reduced = value - test_value - flow_diff_max
    #         if flow_diff_max == 0:
    #             not_reduced_list.append(s_idx)
    #         print(f'server index: {s_idx} prev: {value} \t test value: {test_value} \t flow diff: {flow_diff_max} \t reduced: {reduced}')
    #         if flow_diff_max:
    #             self.values_95_for_s[s_idx] = test_value + flow_diff_max
    #             for g in self.graph4t:
    #                 g.add_edge('s', f's{s_idx}', test_value + flow_diff_max)
            
    #         pass
    #     print(f'after iterate s: {self.values_95_for_s.sum()} \n', sorted(self.values_95_for_s, reverse=True))
    #     for g in self.graph4t:
    #         g.calc_max_flow('s', 't')
    #     flow_sum = np.array(self.max_flow_list).sum()
    #     print(f'final flow sum: {flow_sum}')
        
    # def read_out_network(self):
    #     self.record = self.after_95_record
    #     for t_idx, g in enumerate(self.graph4t):
    #         for s_idx in range(s_len):
    #             for c_idx in range(c_len):
    #                 v = g.get_flow(f's{s_idx}', f'c{c_idx}')
    #                 self.record[t_idx, s_idx, c_idx] += v
    #     self.t_s_record = self.record.sum(axis=-1)

    # def construct_all_graph(self):
    #     t_idx_95 = np.argpartition(self.t_s_record, self.idx_95, axis=0)[self.idx_95]
    #     self.values_95_for_s = self.t_s_record[t_idx_95, np.arange(s_len)]
    #     print(f'95% sum at graph: {np.sum(self.values_95_for_s)}')
    #     self.graph4t: List[Graph] = [ self.construct_graph(t) for t in range(t_len) ]
    #     for t in range(t_len):
    #         self.graph4t[t].calc_max_flow('s', 't')
    #     self.max_flow_list = [ g.max_flow for g in self.graph4t ]
    #     self.flow_sum = np.array(self.max_flow_list).sum()
    #     print(f'flow sum: {np.array(self.max_flow_list).sum()}')
    #     print(f'separate flow: \n {sorted(self.max_flow_list, reverse=True)}')
    #     self.after_95_t_4_s
    
    # def construct_graph(self, t) -> Graph:  # when t = t_len is gather point
    #     graph = Graph()
    #     graph.add([ f's{s}' for s in range(s_len) ])  # server
    #     graph.add([ f'c{c}' for c in range(c_len) ])  # client
    #     graph.add(['s', 't'])
    #     for s in range(s_len):
    #         graph.add_edge('s', f's{s}', self.values_95_for_s[s])
    #     for s, t_set in enumerate(self.after_95_t_4_s):
    #         if t in t_set: graph.add_edge(f's', f's{s}', 0)
    #     for s in range(s_len):
    #         for c in self.qos_avail_for_s[s]:
    #             graph.add_edge(f's{s}', f'c{c}', min(self.demand_after_5_dispatch[t, c], bandwidth[s]))
    #     for c in range(c_len):
    #         graph.add_edge(f'c{c}', f't', self.demand_after_5_dispatch[t, c])
    #     return graph
    
    def dispatch(self):
        my_client_demand = client_demand.copy()
        after_95_dispatched_t_4_s = [ set() for _ in range(s_len) ]
        after_95_dispatched_s_4_t = [ set() for _ in range(t_len) ]
        demand_avg_s = np.zeros((t_len, c_len))
        for c in range(c_len):
            demand_avg_s[:, c] = my_client_demand[:, c] / len(self.qos_avail_for_c[c])
        # dispatch to not dispatched yet for pos > 95%
        while True:
            t_idx, c_idx = self.max_idx_of(demand_avg_s)
            if demand_avg_s[t_idx, c_idx] == 0: break
            need_to_dispatch = my_client_demand[t_idx, c_idx]
            # dispatch to empty server
            s_qos_avail_list = self.qos_avail_for_c[c_idx]
            empty_count = [ self.higher_95_num - len(after_95_dispatched_t_4_s[s_idx]) for s_idx in s_qos_avail_list ]
            remain_for_avail_s = [ bandwidth[s_idx] - self.t_s_record[t_idx, s_idx] for s_idx in s_qos_avail_list ]
            arg = np.lexsort((remain_for_avail_s, empty_count)); arg_idx = -1
            changed = False
            while need_to_dispatch and arg_idx >= -len(s_qos_avail_list) and empty_count[arg[arg_idx]] > 0:
                s_idx = s_qos_avail_list[arg[arg_idx]]
                after_95_dispatched_t_4_s[s_idx].add(t_idx)
                after_95_dispatched_s_4_t[t_idx].add(s_idx)
                need_to_dispatch, assigned = self.assign(t_idx, s_idx, c_idx, need_to_dispatch)
                my_client_demand[t_idx, c_idx] -= assigned
                arg_idx -= 1
                changed = True
            if not changed:
                demand_avg_s[t_idx, c_idx] = 0
                continue
            for c_idx in range(c_len):
                occupied_num = 0
                for s_idx in self.qos_avail_for_c[c_idx]:
                    if t_idx in after_95_dispatched_t_4_s[s_idx]: 
                        occupied_num += 1
                can_dispatch_num = self.qos_avail_num_for_c[c_idx] - occupied_num
                demand_avg_s[t_idx, c_idx] = my_client_demand[t_idx, c_idx] / can_dispatch_num  # TODO: may div by 0
        
        # dispatch the remain
        for (t_idx, c_idx), need_to_dispatch in self.max_idx_gen(my_client_demand): # TODO: need to change client_demand ?
            # max dispatch to pos > 95%
            before_95_s = []
            for s_idx in self.qos_avail_for_c[c_idx]:
                if s_idx in after_95_dispatched_s_4_t[t_idx] or len(after_95_dispatched_t_4_s[s_idx]) < self.higher_95_num:
                    need_to_dispatch, assigned = self.assign(t_idx, s_idx, c_idx, need_to_dispatch)
                    after_95_dispatched_t_4_s[s_idx].add(t_idx)
                    after_95_dispatched_s_4_t[t_idx].add(s_idx)
                    if need_to_dispatch == 0: break
                else:
                    before_95_s.append(s_idx)
            if need_to_dispatch == 0: continue
            # after_95_s = set(after_95_dispatched_s_4_t[t_idx])
            # if after_95_s:
            #     # after_95_remain_bw = [ bandwidth[s_idx] - self.t_s_record[t_idx, s_idx] for s_idx in after_95_s ]
            #     for s_idx in after_95_s.intersection(set(self.qos_avail_for_c[c_idx])):
            #         need_to_dispatch, assigned = self.assign(t_idx, s_idx, c_idx, need_to_dispatch)
            #         if need_to_dispatch == 0: break
            #     if need_to_dispatch == 0: continue

            # avg dispatch to pos < 95%  # TODO: can dispatch to the same level in server

            # before_95_s = list(set(self.qos_avail_for_c[c_idx]) - set(after_95_s))
            avg = need_to_dispatch // len(before_95_s)  # can avg dispatch to server
            remain = need_to_dispatch - avg * len(before_95_s)
            for s_idx in before_95_s:
                if need_to_dispatch == 0: break
                remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
                need_to_dispatch -= assigned
                if need_to_dispatch == 0: break
            if need_to_dispatch:
                raise BaseException('not dispatch all')
    
    def dispatch_old(self):
        self.server_5_t_idx = [ set() for _ in range(len(sname)) ]
        self.server_5_value = [ defaultdict(int) for _ in range(len(sname)) ]
        for (t_idx, c_idx), demand in self.max_idx_gen(client_demand):
            s_list = list(self.qos_avail_for_c[c_idx])
            occu_5_num = []
            occu_5_num = [ len(self.server_5_t_idx[s_idx])-(t_idx in self.server_5_t_idx[s_idx]) for s_idx in s_list ]
            arg = np.argsort(np.array(occu_5_num))
            s_arr = np.array(s_list)[arg]
            for idx, s_idx in enumerate(s_arr):
                if t_idx in self.server_5_t_idx[s_idx]: # in server top 5, put all the resources into
                    if self.server_5_value[s_idx][t_idx] == bandwidth[s_idx]: # server is full at current time, next loop
                        if demand: raise BaseException("1")
                        continue
                    else: # server is not full, try to fill it to full
                        demand, assigned = self.assign(t_idx, s_idx, c_idx, demand)
                        self.server_5_value[s_idx][t_idx] += assigned
                        if demand == 0: break
                    if demand: raise BaseException("2")
                elif len(self.server_5_t_idx[s_idx]) != self.higher_95_num: # not in server top 5, top 5 is not full, fill a blank
                    if self.server_5_value[s_idx][t_idx] == bandwidth[s_idx]: # server is full at current time, next loop
                        continue
                    self.server_5_t_idx[s_idx].add(t_idx)
                    demand, assigned = self.assign(t_idx, s_idx, c_idx, demand)
                    self.server_5_value[s_idx][t_idx] += assigned
                    if demand: raise BaseException("3")
                    if demand == 0: break
                else:  # not in top 5, top 5 is full, put average in all the avail
                    avg_s_arr = s_arr[idx:]
                    avg_dispatch = math.floor(demand / len(avg_s_arr))
                    remain = demand - avg_dispatch * len(avg_s_arr)
                    for ss_idx in avg_s_arr:
                        remain, assigned = self.assign(t_idx, ss_idx, c_idx, avg_dispatch + remain)
                        if remain == 0: break
                    if remain:
                        for ss_idx in avg_s_arr:
                            remain, _ = self.assign(t_idx, ss_idx, c_idx, remain)
                            if remain == 0: break
                    if remain: raise BaseException("dispatch fail, has remain")
                    break
            if demand:
                raise BaseException("dispatch fail")
    
    def _avg_to_each(self, max_can_dispatch, need_dispatch, before_95_list):
        avg = need_dispatch // len(before_95_list)
        remain = need_dispatch - avg * len(before_95_list)
        # dispatch_to_each = np.ones(s_len, dtype=np.int32) * avg
        dispatch_to_each = np.zeros(s_len, dtype=np.int32)
        for s_idx in before_95_list:
            dispatch_to_each[s_idx] = avg
        i = 0
        while remain:
            idx = before_95_list[i]
            dispatch_to_each[idx] += 1
            i += 1
            remain -= 1
        can_not_dispatch_value = 0
        can_dispatch_bool = np.zeros(s_len, dtype=bool)
        for s_idx in before_95_list:
            can_dispatch_bool[s_idx] = True
        for s_idx in before_95_list:
            if dispatch_to_each[s_idx] > max_can_dispatch[s_idx]:
                can_not_dispatch_value += dispatch_to_each[s_idx] - max_can_dispatch[s_idx]
                dispatch_to_each[s_idx] = max_can_dispatch[s_idx]
                can_dispatch_bool[s_idx] = False
        while can_not_dispatch_value:
            can_dispatch_cnt = can_dispatch_bool.sum()
            avg = can_not_dispatch_value // can_dispatch_cnt
            remain = can_not_dispatch_value - avg * can_dispatch_cnt
            for s_idx in before_95_list:
                if can_dispatch_bool[s_idx]:
                    dispatch_to_each[s_idx] += avg
                    if remain:
                        dispatch_to_each[s_idx] += 1
                        remain -= 1
            can_not_dispatch_value = 0
            for s_idx in before_95_list:
                if dispatch_to_each[s_idx] > max_can_dispatch[s_idx]:
                    can_not_dispatch_value += dispatch_to_each[s_idx] - max_can_dispatch[s_idx]
                    dispatch_to_each[s_idx] = max_can_dispatch[s_idx]
                    can_dispatch_bool[s_idx] = False
        return dispatch_to_each
        # avg = need_dispatch // len(before_95_list)
        # remain = need_dispatch - avg * len(before_95_list)
        # # dispatch_to_each = np.ones(s_len, dtype=np.int32) * avg
        # dispatch_to_each = np.zeros(s_len, dtype=np.int32)
        # for s_idx in before_95_list:
        #     dispatch_to_each[s_idx] = avg
        # i = 0
        # while remain:
        #     idx = before_95_list[i]
        #     dispatch_to_each[idx] += 1
        #     i += 1
        #     remain -= 1
        # can_not_dispatch_value = 0
        # can_dispatch_bool = np.zeros(s_len, dtype=bool)
        # for s_idx in before_95_list:
        #     can_dispatch_bool[s_idx] = True
        # for s_idx in before_95_list:
        #     if dispatch_to_each[s_idx] > max_can_dispatch[s_idx]:
        #         can_not_dispatch_value += dispatch_to_each[s_idx] - max_can_dispatch[s_idx]
        #         can_dispatch_bool[s_idx] = False
        # while can_not_dispatch_value:
        #     can_dispatch_cnt = can_dispatch_bool.sum()
        #     avg = can_not_dispatch_value // can_dispatch_cnt
        #     remain = can_not_dispatch_value - avg * can_dispatch_cnt
        #     for s_idx in before_95_list:
        #         if can_dispatch_bool[s_idx]:
        #             dispatch_to_each[s_idx] += avg
        #             if remain:
        #                 dispatch_to_each[s_idx] += 1
        #                 remain -= 1
        #     can_not_dispatch_value = 0
        #     for s_idx in before_95_list:
        #         if dispatch_to_each[s_idx] > max_can_dispatch[s_idx]:
        #             can_not_dispatch_value += dispatch_to_each[s_idx] - max_can_dispatch[s_idx]
        #             can_dispatch_bool[s_idx] = False
        # return dispatch_to_each

    def dispatch_t_avg(self):
        demand_avg = np.zeros(t_len, dtype=np.float32)
        demand = client_demand.copy()
        for t_idx in range(t_len):
            s_set = set()
            for c_idx, c_demand in enumerate(demand[t_idx]):
                if c_demand: s_set.update(self.qos_avail_for_c_set[c_idx])
            demand_avg[t_idx] = demand[t_idx].sum() / len(s_set)
        ts_assign = [ set() for _ in range(t_len)]
        st_assign = [ set() for _ in range(s_len)]
        qos_bool = qos < qos_lim
        ts_visited = np.zeros((t_len, s_len), dtype=bool)
        while True:
            t_idx = np.argmax(demand_avg)
            if demand_avg[t_idx] == 0:
                break
            s_sum_at_t = demand[t_idx] @ qos_bool.T
            idx = -1
            s_idx = np.argsort(s_sum_at_t)[idx]
            while len(st_assign[s_idx]) == self.higher_95_num and idx > -self.higher_95_num:
                idx -= 1
                s_idx = np.argsort(s_sum_at_t)[idx]
            if ts_visited[t_idx, s_idx]:
                demand_avg[t_idx] = 0
                continue
            if len(st_assign[s_idx]) >= self.higher_95_num:
                demand_avg[t_idx] = 0
                continue
            ############## dispatch all ###############
            for c_idx in self.qos_avail_for_s[s_idx]:
                left, assigned = self.assign(t_idx, s_idx, c_idx, demand[t_idx, c_idx])
                if assigned:
                    st_assign[s_idx].add(t_idx)
                    ts_assign[t_idx].add(s_idx)
                    demand[t_idx, c_idx] -= assigned
            ############## dispatch avg ###############
            # ratio = demand[t_idx][self.qos_avail_for_s[s_idx]] / demand[t_idx][self.qos_avail_for_s[s_idx]].sum()
            # left = 0
            # max_to_dispatch = bandwidth[s_idx] - self.t_s_record[t_idx, s_idx]
            # for i, c_idx in enumerate(self.qos_avail_for_s[s_idx]):
            #     left, assigned = self.assign(t_idx, s_idx, c_idx, math.ceil(max_to_dispatch * ratio[i]) + left)
            #     if assigned:
            #         st_assign[s_idx].add(t_idx)
            #         ts_assign[t_idx].add(s_idx)
            #         demand[t_idx, c_idx] -= assigned
            # if left:
            #     for c_idx in self.qos_avail_for_s[s_idx]:
            #         left, assigned = self.assign(t_idx, s_idx, c_idx, left)
            #         if assigned:
            #             st_assign[s_idx].add(t_idx)
            #             ts_assign[t_idx].add(s_idx)
            #             demand[t_idx, c_idx] -= assigned
            #         if left == 0: break
            ############## dispatch end ###############
            s_set = set()
            for c_idx, c_demand in enumerate(demand[t_idx]):
                if c_demand: s_set.update(self.qos_avail_for_c_set[c_idx])
            s_set = s_set - ts_assign[t_idx]
            ts_visited[t_idx, s_idx] = True
            if len(s_set) == 0:
                demand_avg[t_idx] = 0
            else:
                demand_avg[t_idx] = demand[t_idx].sum() / len(s_set)
        # to fill after 95
        for (t_idx, c_idx), need_dispatch in self.max_idx_gen(demand):
            if need_dispatch == 0: break
            for s_idx in self.qos_avail_for_c[c_idx]:
                if t_idx not in st_assign[s_idx] and len(st_assign[s_idx]) == self.higher_95_num:
                # if s_idx not in ts_assign[t_idx] and len(st_assign[s_idx]) == self.higher_95_num: 
                    continue
            # for s_idx in ts_assign[t_idx]:
                # if c_idx not in self.qos_avail_for_s_set[s_idx]: continue
                need_dispatch, assigned = self.assign(t_idx, s_idx, c_idx, need_dispatch)
                st_assign[s_idx].add(t_idx)
                ts_assign[t_idx].add(s_idx)
                demand[t_idx, c_idx] -= assigned
                if need_dispatch == 0: break
        ############# avg diapatch to 95% ###################3
        barrier = np.zeros(s_len, dtype=np.int32)
        for (t_idx, c_idx), need_dispatch in self.max_idx_gen(demand):
            if need_dispatch == 0: break
            # dispatch to max
            before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
            for s_idx in before_95_set:
                can_dispatch = max(barrier[s_idx] - self.t_s_record[t_idx, s_idx], 0)
                can_dispatch = min(need_dispatch, can_dispatch)
                if can_dispatch:
                    left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                    demand[t_idx, c_idx] -= assigned
                    need_dispatch -= assigned
                    if need_dispatch == 0: break
            if need_dispatch == 0: continue
            # after dispatch will at 95%
            max_can_dispatch = np.maximum(bandwidth - self.t_s_record[t_idx], 0)
            before_95_list = list(before_95_set)
            dispatch_to_each = self._avg_to_each(max_can_dispatch, need_dispatch, list(before_95_list))
            for s_idx in before_95_list:
                left, assigned = self.assign(t_idx, s_idx, c_idx, dispatch_to_each[s_idx])
                if left: raise BaseException('not dispatch fully')
                if assigned: barrier[s_idx] = self.t_s_record[t_idx, s_idx]
        ################### normal dispatch #############################
        # for (t_idx, c_idx), need_dispatch in self.max_idx_gen(demand):
        #     if need_dispatch == 0: break
        #     before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
        #     avg = need_dispatch // len(before_95_set)
        #     remain = need_dispatch - avg * len(before_95_set)
        #     for s_idx in before_95_set:
        #         remain, assigned = self.assign(t_idx, s_idx, c_idx, avg + remain)
        #         # client_demand -= assigned
        #     if remain:
        #         for s_idx in before_95_set:
        #             remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
        #             if remain == 0: break

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
    
    def dispatch_from_server(self):
        # DEL:
        ts_assign = [ set() for _ in range(t_len)]
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
                if assigned:
                    ts_assign[t_idx].add(s_idx)
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                self.sum_at_5 += assigned  # DEL
                added += assigned
            arr_t_s[t_idx] = self.demand_after_5_dispatch[t_idx].dot(qos_bool_c_s)
            s_full_filled[s_idx] += 1
            cnt += 1
        print(f'matrix used time: {time.time() - st}')
        ############# normal dispatch ##################
        # self.after_95_record = self.record.copy()
        # st = time.time()
        # for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
        #     s_avail_set = self.qos_avail_for_c_set[c_idx]
        #     s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
        #     avg = need_dispatch // len(s_avail_set)
        #     remain = need_dispatch - avg * len(s_avail_set)
        #     for s_idx in s_avail_set:
        #         remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
        #     if remain: 
        #         for s_idx in s_avail_set:
        #             remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
        #     if remain: 
        #         raise BaseException('not fully dispatched')
        # print(f'remain used time: {time.time() - st}')
        ############### avg dispatch #####################
        barrier = np.zeros(s_len, dtype=np.int32)
        # cand_idx = list(product(range(t_len), range(c_len)))
        # shuffle(cand_idx)
        # for t_idx, c_idx in cand_idx:
        #     need_dispatch = self.demand_after_5_dispatch[t_idx, c_idx]
            # if need_dispatch == 0: continue
        for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
            if need_dispatch == 0: break
        # for (t_idx, c_idx) in list(product(range(t_len), range(c_len))):
        #     need_dispatch = self.demand_after_5_dispatch[t_idx, c_idx]
        #     if need_dispatch == 0: continue
            # dispatch to max
            before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
            for s_idx in before_95_set:
                can_dispatch = max(barrier[s_idx] - self.t_s_record[t_idx, s_idx], 0)
                can_dispatch = min(need_dispatch, can_dispatch)
                if can_dispatch:
                    left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                    self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                    need_dispatch -= assigned
                    if need_dispatch == 0: break
            if need_dispatch == 0: continue
            # after dispatch will at 95%
            max_can_dispatch = np.maximum(bandwidth - self.t_s_record[t_idx], 0)
            before_95_list = list(before_95_set)
            dispatch_to_each = self._avg_to_each(max_can_dispatch, need_dispatch, list(before_95_list))
            for s_idx in before_95_list:
                left, assigned = self.assign(t_idx, s_idx, c_idx, dispatch_to_each[s_idx])
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                if left: raise BaseException('not dispatch fully')
                if assigned: barrier[s_idx] = self.t_s_record[t_idx, s_idx]

    def dispatch_from_server_2(self):
        # DEL:
        ts_assign = [ set() for _ in range(t_len)]
        self.sum_at_5 = 0
        s_full_filled = np.zeros(s_len, dtype=np.int32)
        after_95_t_includ_s = defaultdict(set)
        self.demand_after_5_dispatch = client_demand.copy()
        qos_bool_c_s_orig = np.array((qos < qos_lim).T, order='F')
        qos_bool_c_s = qos_bool_c_s_orig
        s_idx_resotre_arr = np.zeros(s_len, dtype=np.int32)
        s_idx_deleted = []
        can_dispatch_num = deepcopy(self.qos_avail_num_for_s)
        for s, v in enumerate(can_dispatch_num):
            if v == 0:
                can_dispatch_num[s] = 1
        arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s_orig / can_dispatch_num   # t * c  dot  c * s  -->  t * s
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
                can_dispatch_to_servver_num = [ 1 for _ in range(s_len) ]
                for c_idx, c_demand in enumerate(self.demand_after_5_dispatch[t_idx]):
                    if c_demand: 
                        for s_idx in self.qos_avail_for_c[c_idx]:
                            can_dispatch_to_servver_num[s_idx] += 1
                can_dispatch_to_servver_num = np.delete(can_dispatch_to_servver_num, s_idx_deleted)
                arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s / can_dispatch_to_servver_num  # t * c  dot  c * s  -->  t * s
                continue
            c_avail_set = self.qos_avail_for_s[s_idx]
            if not c_avail_set: continue
            after_95_t_includ_s[t_idx].add(s_idx)
            self.after_95_t_4_s[s_idx].add(t_idx)
            added = 0; left = 0
            for c_idx in self.qos_avail_for_s[s_idx]: # TODO: select c_idx scheme
                if added == bandwidth[s_idx]: break
                left, assigned = self.assign(t_idx, s_idx, c_idx, self.demand_after_5_dispatch[t_idx, c_idx])
                if assigned:
                    ts_assign[t_idx].add(s_idx)
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                self.sum_at_5 += assigned  # DEL
                added += assigned
            can_dispatch_to_servver_num = [ 1 for _ in range(s_len) ]
            for c_idx, c_demand in enumerate(self.demand_after_5_dispatch[t_idx]):
                if c_demand: 
                    for s_idx in self.qos_avail_for_c[c_idx]:
                        can_dispatch_to_servver_num[s_idx] += 1
            can_dispatch_to_servver_num = np.delete(can_dispatch_to_servver_num, s_idx_deleted)
            arr_t_s[t_idx] = self.demand_after_5_dispatch[t_idx].dot(qos_bool_c_s) / can_dispatch_to_servver_num
            s_full_filled[s_idx] += 1
            cnt += 1
        print(f'matrix used time: {time.time() - st}')
        ############# normal dispatch ##################
        # self.after_95_record = self.record.copy()
        # st = time.time()
        # for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
        #     s_avail_set = self.qos_avail_for_c_set[c_idx]
        #     s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
        #     avg = need_dispatch // len(s_avail_set)
        #     remain = need_dispatch - avg * len(s_avail_set)
        #     for s_idx in s_avail_set:
        #         remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
        #     if remain: 
        #         for s_idx in s_avail_set:
        #             remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
        #     if remain: 
        #         raise BaseException('not fully dispatched')
        # print(f'remain used time: {time.time() - st}')
        ############### avg dispatch #####################
        barrier = np.zeros(s_len, dtype=np.int32)
        # cand_idx = list(product(range(t_len), range(c_len)))
        # shuffle(cand_idx)
        # for t_idx, c_idx in cand_idx:
        #     need_dispatch = self.demand_after_5_dispatch[t_idx, c_idx]
            # if need_dispatch == 0: continue
        for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
            if need_dispatch == 0: break
            # dispatch to max
            before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
            for s_idx in before_95_set:
                can_dispatch = max(barrier[s_idx] - self.t_s_record[t_idx, s_idx], 0)
                can_dispatch = min(need_dispatch, can_dispatch)
                if can_dispatch:
                    left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                    self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                    need_dispatch -= assigned
                    if need_dispatch == 0: break
            if need_dispatch == 0: continue
            # after dispatch will at 95%
            max_can_dispatch = np.maximum(bandwidth - self.t_s_record[t_idx], 0)
            before_95_list = list(before_95_set)
            dispatch_to_each = self._avg_to_each(max_can_dispatch, need_dispatch, list(before_95_list))
            for s_idx in before_95_list:
                left, assigned = self.assign(t_idx, s_idx, c_idx, dispatch_to_each[s_idx])
                if left: raise BaseException('not dispatch fully')
                if assigned: barrier[s_idx] = self.t_s_record[t_idx, s_idx]

    def dispatch_from_server_3(self):
        # DEL:
        ts_assign = [ set() for _ in range(t_len)]
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
                if assigned:
                    ts_assign[t_idx].add(s_idx)
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                self.sum_at_5 += assigned  # DEL
                added += assigned
            arr_t_s[t_idx] = self.demand_after_5_dispatch[t_idx].dot(qos_bool_c_s)
            s_full_filled[s_idx] += 1
            cnt += 1
        print(f'matrix used time: {time.time() - st}')
        ############# normal dispatch ##################
        # self.after_95_record = self.record.copy()
        # st = time.time()
        # for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
        #     s_avail_set = self.qos_avail_for_c_set[c_idx]
        #     s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
        #     avg = need_dispatch // len(s_avail_set)
        #     remain = need_dispatch - avg * len(s_avail_set)
        #     for s_idx in s_avail_set:
        #         remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
        #     if remain: 
        #         for s_idx in s_avail_set:
        #             remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
        #     if remain: 
        #         raise BaseException('not fully dispatched')
        # print(f'remain used time: {time.time() - st}')
        ############### avg dispatch #####################
        barrier = np.zeros(s_len, dtype=np.int32)
        # cand_idx = list(product(range(t_len), range(c_len)))
        # shuffle(cand_idx)
        # for t_idx, c_idx in cand_idx:
        #     need_dispatch = self.demand_after_5_dispatch[t_idx, c_idx]
            # if need_dispatch == 0: continue
        # for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
        #     if need_dispatch == 0: break
        t_demand = self.demand_after_5_dispatch.sum(axis=-1)
        arg = np.argsort(-t_demand)
        # for t_idx in arg:
        for t_idx in range(t_len):
            for c_idx in range(c_len):
                need_dispatch = self.demand_after_5_dispatch[t_idx, c_idx]
                if need_dispatch:
                    # dispatch to max
                    before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
                    for s_idx in before_95_set:
                        can_dispatch = max(barrier[s_idx] - self.t_s_record[t_idx, s_idx], 0)
                        can_dispatch = min(need_dispatch, can_dispatch)
                        if can_dispatch:
                            left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                            self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                            need_dispatch -= assigned
                            if need_dispatch == 0: break
            for c_idx in range(c_len):
                need_dispatch = self.demand_after_5_dispatch[t_idx, c_idx]
                if need_dispatch == 0: continue
                # after dispatch will at 95%
                max_can_dispatch = np.maximum(bandwidth - self.t_s_record[t_idx], 0)
                before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
                before_95_list = list(before_95_set)
                dispatch_to_each = self._avg_to_each(max_can_dispatch, need_dispatch, list(before_95_list))
                for s_idx in before_95_list:
                    left, assigned = self.assign(t_idx, s_idx, c_idx, dispatch_to_each[s_idx])
                    self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                    if left: raise BaseException('not dispatch fully')
                    if assigned: barrier[s_idx] = self.t_s_record[t_idx, s_idx]
    
    def dispatch_from_server_5time(self):
        ts_assign = [ set() for _ in range(t_len)]
        st_assign = [ set() for _ in range(s_len)]
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
        ############ dispatch to 5% ######################
        cnt = 0
        for _ in range(s_len):
            cnt += 1
            t_idx, s_idx = self.max_idx_of(arr_t_s)
            t_series = arr_t_s[:, s_idx]
            s_idx = self._restore_idx(s_idx, s_idx_resotre_arr)
            t_arg = np.argpartition(t_series, -self.higher_95_num)[-self.higher_95_num:]
            for t_idx in t_arg:
                need_dispatch = t_series[t_idx]
                for c_idx in self.qos_avail_for_s[s_idx]:
                    can_dispatch = min(need_dispatch, self.demand_after_5_dispatch[t_idx, c_idx])
                    left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                    if assigned:
                        self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                        ts_assign[t_idx].add(s_idx)
                        st_assign[s_idx].add(t_idx)
                        need_dispatch -= assigned
                    if need_dispatch == 0: break
            s_idx_deleted.append(s_idx)
            s_idx_resotre_arr[s_idx] = 1
            qos_bool_c_s = np.delete(qos_bool_c_s_orig, s_idx_deleted, axis=1)
            arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s  # t * c  dot  c * s  -->  t * s
        print(f'matrix used time: {time.time() - st}')
        ############# dispatch to prev 95% ###################
        ############# normal dispatch ##################
        # self.after_95_record = self.record.copy()
        # st = time.time()
        # for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
        #     s_avail_set = self.qos_avail_for_c_set[c_idx]
        #     s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
        #     avg = need_dispatch // len(s_avail_set)
        #     remain = need_dispatch - avg * len(s_avail_set)
        #     for s_idx in s_avail_set:
        #         remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
        #     if remain: 
        #         for s_idx in s_avail_set:
        #             remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
        #     if remain: 
        #         raise BaseException('not fully dispatched')
        ############### avg dispatch #####################
        barrier = np.zeros(s_len, dtype=np.int32)
        for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
            if need_dispatch == 0: break
            # dispatch to max
            before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
            for s_idx in before_95_set:
                can_dispatch = max(barrier[s_idx] - self.t_s_record[t_idx, s_idx], 0)
                can_dispatch = min(need_dispatch, can_dispatch)
                if can_dispatch:
                    left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                    self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                    need_dispatch -= assigned
                    if need_dispatch == 0: break
            if need_dispatch == 0: continue
            # after dispatch will at 95%
            max_can_dispatch = np.maximum(bandwidth - self.t_s_record[t_idx], 0)
            before_95_list = list(before_95_set)
            dispatch_to_each = self._avg_to_each(max_can_dispatch, need_dispatch, list(before_95_list))
            for s_idx in before_95_list:
                left, assigned = self.assign(t_idx, s_idx, c_idx, dispatch_to_each[s_idx])
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                if left: raise BaseException('not dispatch fully')
                if assigned: barrier[s_idx] = self.t_s_record[t_idx, s_idx]
        print(f'remain used time: {time.time() - st}')

    def dispatch_from_server_5time_2(self):
        ts_assign = [ set() for _ in range(t_len)]
        st_assign = [ set() for _ in range(s_len)]
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
        ############ dispatch to 5% ######################
        s_idx_arg = np.argsort(-bandwidth)
        for s_idx in s_idx_arg:
            t_series = arr_t_s[:, s_idx]
            # s_idx = self._restore_idx(s_idx, s_idx_resotre_arr)
            t_arg = np.argpartition(t_series, -self.higher_95_num)[-self.higher_95_num:]
            for t_idx in t_arg:
                need_dispatch = t_series[t_idx]
                for c_idx in self.qos_avail_for_s[s_idx]:
                    can_dispatch = min(need_dispatch, self.demand_after_5_dispatch[t_idx, c_idx])
                    left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                    if assigned:
                        self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                        ts_assign[t_idx].add(s_idx)
                        st_assign[s_idx].add(t_idx)
                        need_dispatch -= assigned
                    if need_dispatch == 0: break
            # s_idx_deleted.append(s_idx)
            # s_idx_resotre_arr[s_idx] = 1
            # qos_bool_c_s = np.delete(qos_bool_c_s_orig, s_idx_deleted, axis=1)
            arr_t_s = self.demand_after_5_dispatch @ qos_bool_c_s  # t * c  dot  c * s  -->  t * s
        print(f'matrix used time: {time.time() - st}')
        ############# dispatch to prev 95% ###################
        ############# normal dispatch ##################
        # self.after_95_record = self.record.copy()
        # st = time.time()
        # for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
        #     s_avail_set = self.qos_avail_for_c_set[c_idx]
        #     s_avail_set = s_avail_set - after_95_t_includ_s[t_idx]
        #     avg = need_dispatch // len(s_avail_set)
        #     remain = need_dispatch - avg * len(s_avail_set)
        #     for s_idx in s_avail_set:
        #         remain, assigned = self.assign(t_idx, s_idx, c_idx, remain + avg)
        #     if remain: 
        #         for s_idx in s_avail_set:
        #             remain, assigned = self.assign(t_idx, s_idx, c_idx, remain)
        #     if remain: 
        #         raise BaseException('not fully dispatched')
        ############### avg dispatch #####################
        barrier = np.zeros(s_len, dtype=np.int32)
        for (t_idx, c_idx), need_dispatch in self.max_idx_gen(self.demand_after_5_dispatch):
            if need_dispatch == 0: break
        # tc_list = list(product(range(t_len), range(c_len)))
        # shuffle(tc_list)
        # for t_idx, c_idx in tc_list:
        #     need_dispatch = self.demand_after_5_dispatch[t_idx, c_idx]
        #     if need_dispatch == 0: continue
            # dispatch to max
            before_95_set = self.qos_avail_for_c_set[c_idx] - ts_assign[t_idx]
            for s_idx in before_95_set:
                can_dispatch = max(barrier[s_idx] - self.t_s_record[t_idx, s_idx], 0)
                can_dispatch = min(need_dispatch, can_dispatch)
                if can_dispatch:
                    left, assigned = self.assign(t_idx, s_idx, c_idx, can_dispatch)
                    self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                    need_dispatch -= assigned
                    if need_dispatch == 0: break
            if need_dispatch == 0: continue
            # after dispatch will at 95%
            max_can_dispatch = np.maximum(bandwidth - self.t_s_record[t_idx], 0)
            before_95_list = list(before_95_set)
            dispatch_to_each = self._avg_to_each(max_can_dispatch, need_dispatch, list(before_95_list))
            for s_idx in before_95_list:
                left, assigned = self.assign(t_idx, s_idx, c_idx, dispatch_to_each[s_idx])
                self.demand_after_5_dispatch[t_idx, c_idx] -= assigned
                if left: raise BaseException('not dispatch fully')
                if assigned: barrier[s_idx] = self.t_s_record[t_idx, s_idx]
        print(f'remain used time: {time.time() - st}')

if __name__ == '__main__':
    get_data()
    start_time = time.time()
    s = Solution()
    # s.dispatch()
    # s.dispatch_t_avg()
    s.dispatch_from_server_5time_2()
    if LOCAL: 
        print(f'used time normal: {(time.time()-start_time):.2f}')
        s.calc_score95(False)

    # s.construct_all_graph()
    # s.iterate_s_cap()
    # s.read_out_network()
        
    # print(f'demand except 5% is {s.demand_after_5_dispatch.ravel().sum()}')
    # print(f'sum at 5: {s.sum_at_5}, demand - sum_at_5: {client_demand.ravel().sum()-s.sum_at_5}')
    # s.iterate_s_cap()
    
    # s.analyse_larger_than_95()

    if LOCAL: 
        s.check_output_valid()
        s.calc_score95(True)
        time_threshould = 10
    else: 
        time_threshould = 283

    prev_score = s.calc_score95(print_sep=False)
    while time.time() - start_time < time_threshould:
        # s.dispatch_again()
        # s.dispatch_again_batch_for_one_server(i)
        s.dispatch_again()
        curr_score = s.calc_score95(print_sep=False)
        # if (prev_score - curr_score) / curr_score < 0.000000001: 
        #     break
        prev_score = curr_score

    s.output()
    if LOCAL: 
        print(f'used time: {(time.time()-start_time):.2f}')
        s.check_output_valid()