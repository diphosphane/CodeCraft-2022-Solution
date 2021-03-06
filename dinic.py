import numpy as np
from collections import defaultdict
#Dinic Algorithm
class Dinic():
    def __init__(self) -> None:
        self.cnt = 0
        self.str2int = {}
        self.adjacent_table = {}
        self.forward_matrix = None
        # self.edge = {}
    
    def add(self, vertices):
        for vertex in vertices:
            new_d = {}
            new_d[self.cnt] = defaultdict(int)
            self.adjacent_table.update(new_d)
            self.str2int[vertex] = self.cnt
            self.cnt += 1

    def add_edge(self, source, terminal, capacity):
        # self.edge.append((source, terminal, capacity))
        neighbor_dict = self.adjacent_table[self.str2int[source]]
        neighbor_dict[self.str2int[terminal]] = capacity

    def construct_table(self):
        num = len(self.str2int)
        self.adjacent_table = np.zeros((num, num), dtype=np.int32)
        for s, t, c in self.edge:
            ss, tt = self.str2int[s], self.str2int[t]
            self.adjacent_table[ss, tt] = c
    
    def calc_max_flow(self, source, terminal):
        # if not self.adjacent_table:
        #     self.construct_table()
        self.max_flow = self._calc_max_flow(self.adjacent_table, self.str2int[source], self.str2int[terminal])

    def bfs(self, C, F, s, t):
        n = len(C)
        queue = []
        queue.append(s)
        global level
        level = n * [0]  # initialization
        level[s] = 1  
        while queue:
            k = queue.pop(0)
            # for i, cap in F[k].items():
            #     if (F[k].get(i, 0) < cap) and (level[i] == 0): # not visited
            #         level[i] = level[k] + 1
            #         queue.append(i)
            for i in range(n):
                if (F[k][i] < C[k][i]) and (level[i] == 0): # not visited
                    level[i] = level[k] + 1
                    queue.append(i)
        return level[t] > 0

    def dfs(self, C, F, k, cp):
        tmp = cp
        if k == len(C)-1:
            return cp
        for i in range(len(C)):
            if (level[i] == level[k] + 1) and (F[k][i] < C[k][i]):
                f = self.dfs(C,F,i,min(tmp,C[k][i] - F[k][i]))
                F[k][i] = F[k][i] + f
                F[i][k] = F[i][k] - f
                tmp = tmp - f
        # for i in range(len(C)):
        #     if (level[i] == level[k] + 1) and (F[k][i] < C[k][i]):
        #         f = Dfs(C,F,i,min(tmp,C[k][i] - F[k][i]))
        #         F[k][i] = F[k][i] + f
        #         F[i][k] = F[i][k] - f
        #         tmp = tmp - f
        return cp - tmp

    def _calc_max_flow(self, C, s, t):
        F = np.zeros((self.cnt, self.cnt), dtype=np.int32)
        # F = defaultdict(defaultdict(int))
        flow = 0
        while(self.bfs(C,F,s,t)):
            flow = flow + self.dfs(C,F,s,100000)
        self.forward_matrix = F
        return flow
    
    def get_flow(self, source, terminal):
        s, t = self.str2int[source], self.str2int[terminal]
        return self.forward_matrix[s][t]
