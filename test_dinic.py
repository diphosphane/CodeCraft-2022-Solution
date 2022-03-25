from dinic_git import Dinic as Graph

g = Graph()
g.add([ str(i) for i in range(1, 13) ])
g.add_edge('1', '2', 8)
g.add_edge('1', '3', 6)
g.add_edge('2', '5', 4)
g.add_edge('2', '4', 4)
g.add_edge('3', '4', 3)
g.add_edge('3', '9', 3)
g.add_edge('4', '8', 3)
g.add_edge('4', '7', 6)
g.add_edge('5', '6', 4)
g.add_edge('6', '7', 1)
g.add_edge('6', '10', 3)
g.add_edge('7', '8', 7)
g.add_edge('8', '9', 7)
g.add_edge('8', '11', 4)
g.add_edge('9', '12' , 10)
g.add_edge('10', '11', 3)
g.add_edge('11', '12', 5)
g.calc_max_flow('1', '12')
print(1)
print(g.get_flow('7', '9'))
print(g.get_flow('4', '7'))
print(g.get_flow('11', '12'))