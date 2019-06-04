from collections import defaultdict, deque

class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance


def dijkstra(graph, origin, destination):
    visited = {origin: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            try:
                weight = current_weight + graph.distances[(min_node, edge)]
            except:
                continue
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    # backtract the shortest path
    cur = path[destination]
    full_path = deque()

    while cur != origin:
        full_path.appendleft(cur)
        cur = path[cur]

    full_path.appendleft(origin)
    full_path.append(destination)
    full_path = list(full_path)

    # calculate cost along path
    costs = []
    for idx in range(len(full_path) - 1):
        costs.append(graph.distances[(full_path[idx], full_path[idx+1])])

    return costs, full_path

if __name__ == '__main__':
    # test dijkstra
    graph = Graph()
    for node in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        graph.add_node(node)
    graph.add_edge('A', 'B', 10)
    graph.add_edge('A', 'C', 20)
    graph.add_edge('B', 'D', 15)
    graph.add_edge('C', 'D', 30)
    graph.add_edge('B', 'E', 50)
    graph.add_edge('D', 'E', 30)
    graph.add_edge('E', 'F', 5)
    graph.add_edge('F', 'G', 2)
    print(dijkstra(graph, 'A', 'D')) # output: (25, ['A', 'B', 'D']) 