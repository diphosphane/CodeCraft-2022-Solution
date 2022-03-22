# -*- coding: utf-8 -*-
# Copyright (c) 2019 - Youpeng Hu <yoooooohu@foxmail.com>
import copy

class Graph:
    def __init__(self):
        self.adjacent_table = {}
    
    def add(self, vertices):
        for vertex in vertices:
            self.adjacent_table.update({vertex: {}})

    def add_edge(self, source, terminal, capacity):
        neighbor_dict = self.adjacent_table[source]
        neighbor_dict[terminal] = capacity

    def max_capacity_augment(self, source, terminal):
        # print('###################################')
        # print('Max Capacity Augmentation algorithm')
        self.f_table = {}
        self.r_table = copy.deepcopy(self.adjacent_table)

        path = self.find_max_bottleneck_DFS(source, terminal, 0, {})        
        while path:
            # print('path', end=" >>>>>>>>>>>>>>\n")
            # print(path)
            bottleneck = self.found_bottleneck(path, terminal)

            self.augment_flow(bottleneck, path, terminal)

            path = self.find_max_bottleneck_DFS(source, terminal, 0, {})

        return self.f_table

    def find_max_bottleneck_DFS(self, source, terminal, level, level_table, max_bottleneck = 1000):
        # print('(source, terminal, max_bottleneck)', end=" ->\n")
        # print((source, terminal, max_bottleneck))

        level += 1
        for end, weight in self.r_table[source].items():
            if weight > 0:
                if level not in level_table:
                    level_table[level] = {}
                if end not in level_table[level]:
                    level_table[level][end] = {}    
                level_table[level][end] = {source: weight}
                if end == terminal:
                    return level_table
                if weight >= max_bottleneck:
                    return self.find_max_bottleneck_DFS(end, terminal, level, level_table, max_bottleneck = max_bottleneck)
                elif weight < max_bottleneck:
                    return self.find_max_bottleneck_DFS(end, terminal, level, level_table, max_bottleneck = weight)
            elif weight < 0:
                raise ("the value of weigh have some problem")

    def calc_max_flow(self, source, terminal):
        # print('###################################')
        # print('edmondKarp algorithm')
        self.f_table = {}
        self.r_table = copy.deepcopy(self.adjacent_table)
        path = self.find_terminal_BFS([source], terminal, 0, {}, [source])        
        while path:
            # print('argument table', end=" >>>>>>>>>>>>>>\n")
            # print(path)
            bottleneck = self.found_bottleneck(path, terminal)

            self.augment_flow(bottleneck, path, terminal)

            path = self.find_terminal_BFS([source], terminal, 0, {}, [source])
  
        return self.f_table

    def find_terminal_BFS(self, start_list, terminal, level, level_table, visited):
        # print('(start_list, terminal, level, level_table, visited)', end=" ->\n")
        # print((start_list, terminal, level, level_table, visited))
        level += 1
        end_list = []
        for start in start_list:
            for end, weight in self.r_table[start].items():
                if (weight > 0) & (end not in visited):
                    if level not in level_table:
                        level_table[level] = {}
                    if end not in level_table[level]:
                        level_table[level][end] = {}
                    level_table[level][end] = {start: weight}
                    if terminal == end:
                        return level_table
                    visited.append(end)
                    end_list.append(end)

                elif weight < 0:
                    raise ("the value of weight have some problem")
        if end_list:
            return self.find_terminal_BFS(end_list, terminal, level, level_table, visited)

    def found_bottleneck(self, path, terminal):
        weight_list = []
        tmp_level = max(path.keys())
        # print('tmp_level', end =" ->")
        # print(tmp_level)

        start = terminal
        while tmp_level > 0:
            for start, weight in path[tmp_level][start].items():
                pass
            tmp_level -= 1
            weight_list.append(weight)                
        # print('Bottleneck of traffic -> {}'.format(min(weight_list)))
        return min(weight_list)

    def augment_flow(self, bottleneck, path, terminal):
        tmp_level = max(path.keys())

        end = terminal
        # print('path:\n', terminal, end="")
        while tmp_level > 0:
            for start, weight in path[tmp_level][end].items():
                pass
            # print("->", start, weight, end="")
            if start not in self.f_table:
                self.f_table[start] = {}           
            if end not in self.f_table[start]:
                self.f_table[start][end] = 0          
            self.f_table[start][end] += bottleneck
            self.r_table[start][end] -= bottleneck
            tmp_level -= 1
            end = start
        # print()     
        # print('self.r_table', end=" ->\n")
        # print(self.r_table)
        # print('self.f_table', end=" ->\n")
        # print(self.f_table) 
    
    @property
    def max_flow(self):
        flow = 0
        for f, t_dict in self.f_table.items():
            for t, v in t_dict.items():
                if t == 't': flow += v
        return flow


if __name__ == '__main__':
    graph = Graph()
    graph.add(['s','a','b','c','d','e','f','g','h','i','j','t'])
    graph.add_edge('s', 'a', 6)
    graph.add_edge('s', 'c', 8)
    graph.add_edge('a', 'b', 3)
    graph.add_edge('a', 'd', 3)
    graph.add_edge('b', 't', 10)
    graph.add_edge('c', 'd', 4)
    graph.add_edge('c', 'f', 4)
    graph.add_edge('d', 'e', 3)
    graph.add_edge('d', 'g', 6)
    graph.add_edge('e', 'b', 7)
    graph.add_edge('e', 'j', 4)
    graph.add_edge('f', 'h', 4)
    graph.add_edge('g', 'e', 7)
    graph.add_edge('h', 'g', 1)
    graph.add_edge('h', 'i', 3)
    graph.add_edge('i', 'j', 3)
    graph.add_edge('j', 't', 5)

    graph.EK_algorithm('s', 't')
    # print('gragh.f_table for Edmond_Karp', end="->>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    print(graph.f_table)
    tmp = 0
    for f, t_dict in graph.f_table.items():
        for t, v in t_dict.items():
            if t == 't':
                tmp += v
    print(f'max flow: {tmp}')
    # print('gragh.r_table for Edmond_Karp', end="->>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    # print(gragh.r_table)   

    # gragh.maxCapacityAugmentation('s', 't')
    # print('gragh.f_table for maxCapacityAugmentation', end="->>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    # print(gragh.f_table)
    # print('gragh.r_table for maxCapacityAugmentation', end="->>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    # print(gragh.r_table)

