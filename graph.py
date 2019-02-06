"""
Simple graph and various search and path algorithms.

A lot of this comes from the excellent article

https://www.redblobgames.com/pathfinding/a-star/introduction.html

"""
from collections import defaultdict, deque
import heapq
import math
from typing import cast, Any, Callable, DefaultDict, Dict, Iterable, Hashable, List, Mapping, Set, Tuple

class GraphException(Exception):
    pass


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[int, Hashable]] = []
    
    def empty(self) -> bool:
        return len(self.elements) == 0
    
    def put(self, item: Hashable, priority: int) -> None:
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> Hashable:
        return heapq.heappop(self.elements)[1]

def taxidistance(target: Tuple[int, int], next: Tuple[int, int]) -> int:
    """
    Common A* heuristic -- the "taxidistance" also known as "Manhattan distance".
    """
    return abs(next[0]-target[0]) + abs(next[1]-target[1])

def nocost(target: Hashable, next: Hashable) -> int:
    return 0

class Graph:
    """
    Simple graph representation. Nodes must be hashable.
    """
    def __init__(self):
        self.nodes: Set[Hashable] = set()
        self.edges = defaultdict(dict)

    def add(self, node: Hashable) -> None:
        self.nodes.add(node)

    def edge(self, start: Hashable, end: Hashable, cost: int = 1) -> None:
        self.edges[start][end] = cost

    def _unwind_path(self, came_from: Dict[Hashable, Any], start: Hashable, end: Hashable) -> List[Hashable]:
        current = end
        path: List[Hashable] = []

        while current != start:
            path.append(current)
            try:
                current = came_from[current]
            except KeyError:
                raise GraphException(f"No path exists from {start} to {end}")

        path.append(start)
        path.reverse()

        return path

    def _unwind_multiple_paths(
            self, 
            came_from: DefaultDict[Hashable, List[Hashable]], 
            start: Hashable, 
            end: Hashable
    ) -> List[List[Hashable]]:

        paths: List[List[Hashable]] = []

        done = False
        while not done:
            current = end
            path = []
            while current != start:
                path.append(current)
                if len(came_from[current]) > 0:
                    current = came_from[current].pop()
                else:
                    done = True
                    break
            if done:
                break
            path.append(start)
            path.reverse()
            paths.append(path)

        return paths

    def from_rows(self, data: List[List[str]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Primarily for testing -- make a grid-like graph based on a simple string notation.
        """
        start: Tuple[int, int] = (0, 0)
        end: Tuple[int, int] = (0, 0)
         
        for y, row in enumerate(data):
            for x, ch in enumerate(row):
                pos = (x, y)
                if ch in {".", "S", "T"}:
                    self.add(pos)
                if ch == "S":
                    start = pos
                elif ch == "T":
                    end = pos

        for node in self.nodes:
            (x, y) = cast(Tuple[int, int], node)
            for n in {(x, y-1), (x-1, y), (x, y+1), (x+1, y)}:
                if n in self.nodes:
                    self.edge((x, y), n)
                    self.edge(n, (x, y))

        return start, end

    def render_path(self, path: Iterable[Hashable], xsize: int, ysize: int) -> List[str]:
        """
        Primarily for testing -- render a path on a grid-like graph based on a simple 
        string notation.
        """
        data: List[str] = []
        for y in range(ysize):
            row = ""
            for x in range(xsize):
                pos = (x, y)
                if pos in path:
                    row += "*"
                elif pos in self.nodes:
                    row += "."
                else:
                    row += "X"
            data.append(row)

        return data

    def breadth_first_search(self, start: Hashable, end: Hashable) -> List[Hashable]:
        """
        Breadth-first search with early exit. Edge cost not accounted for.
        """
        frontier = deque([start])
        came_from: Dict[Hashable, Any] = {start: None}

        while frontier:
            current = frontier.popleft()

            if current == end:
                break
            
            if current in self.edges:
                for (next, _) in self.edges[current].items():
                    if next not in came_from:
                        frontier.append(next)
                        came_from[next] = current

        return self._unwind_path(came_from, start, end)

    def uniform_cost_search(self, start: Hashable, end: Hashable) -> List[Hashable]:
        """
        Dijkstra's uniform cost search finds the lowest-cost path from
        start to end.
        """
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from: Dict[Hashable, Any] = {start: None}
        cost_so_far: Dict[Hashable, int] = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if current == end:
                break

            if current in self.edges:
                for (next, cost) in self.edges[current].items():
                    new_cost = cost_so_far[current] + cost
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost
                        frontier.put(next, priority)
                        came_from[next] = current

        return self._unwind_path(came_from, start, end)

    def a_star_search(
            self, 
            start: Hashable, 
            end: Hashable, 
            heuristic: Callable[[Hashable, Hashable], int] = nocost
    ) -> List[Hashable]:
        """
        A* search. The heuristic should be a function that gives a "guess"
        of how well travelling from a to b takes you towards the target. 
        This could for example be the "manhattan distance" if the graph is 
        looking like a grid:

        def taxidistance(target, next):
            return abs(next[0]-target[0]) + abs(next[1]-target[1])

        As long as the given heuristic does not over-estimate the cost,
        A* converges on the lowest-cost path whilst exploring fewer nodes
        than UCS (Dijkstra) and other graph search algorithms.

        With no heuristic cost, A* is equivalent to Dijkstra
        """
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from: Dict[Hashable, Any] = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if current == end:
                break

            if current in self.edges:
                for (next, cost) in self.edges[current].items():
                    new_cost = cost_so_far[current] + cost
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost + heuristic(end, next)
                        frontier.put(next, priority)
                        came_from[next] = current

        return self._unwind_path(came_from, start, end)

    def uniform_cost_search_all(self, start: Hashable, end: Hashable) -> List[List[Hashable]]:
        """
        Dijkstra's uniform cost search modified to find *all* lowest-cost paths between
        start and end.
        """
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from: DefaultDict[Hashable, List[Hashable]] = defaultdict(list)
        cost_so_far: Dict[Hashable, int] = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            print(current)

            if current == end:
                break

            if current in self.edges:
                for (next, cost) in self.edges[current].items():
                    new_cost = cost_so_far[current] + cost
                    if next not in cost_so_far or new_cost <= cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost
                        frontier.put(next, priority)
                        came_from[next].append(current)

        return self._unwind_multiple_paths(came_from, start, end)

    
                    
    # def uniform_cost_search_all(self, start: Hashable, end: Hashable) -> List[List[Hashable]]:
    #     Item = namedtuple("Item", ["node", "previous", "cost"])
    #     frontier = PriorityQueue()
    #     frontier.put(Item(start, None, 0), 0)

    #     while not frontier.empty():
    #         current = frontier.get()

    #         if current.node == end:
    #             paths = [current]
    #             for item in frontier.elements:
    #                 item = cast(Item, item)
    #                 if (
    #                     item.node == current.node and
    #                     item.cost == current.cost
    #                 ):
    #                     paths.append[item]
    #             break

    #         if current.node in self.edges:
    #             for (next, cost) in self.edges[current.node].items():
    #                 new_cost = current.cost + cost

    #                 for item in frontier.elements:
    #                     item = cast(Item, item)

    #                 if (
    #                     item.cost < cost && RouteNode.node == neighbour) {
    #             frontier.add(new RouteNode(neighbour, current, cost, cost + getEstimate(neighbour, goal));
    #         }
    #     }



# function YenKSP(Graph, source, sink, K):
#    // Determine the shortest path from the source to the sink.
#    A[0] = Dijkstra(Graph, source, sink);
#    // Initialize the set to store the potential kth shortest path.
#    B = [];
   
#    for k from 1 to K:
#        // The spur node ranges from the first node to the next to last node in the previous k-shortest path.
#        for i from 0 to size(A[k − 1]) − 2:
           
#            // Spur node is retrieved from the previous k-shortest path, k − 1.
#            spurNode = A[k-1].node(i);
#            // The sequence of nodes from the source to the spur node of the previous k-shortest path.
#            rootPath = A[k-1].nodes(0, i);
           
#            for each path p in A:
#                if rootPath == p.nodes(0, i):
#                    // Remove the links that are part of the previous shortest paths which share the same root path.
#                    remove p.edge(i,i + 1) from Graph;
           
#            for each node rootPathNode in rootPath except spurNode:
#                remove rootPathNode from Graph;
           
#            // Calculate the spur path from the spur node to the sink.
#            spurPath = Dijkstra(Graph, spurNode, sink);
           
#            // Entire path is made up of the root path and spur path.
#            totalPath = rootPath + spurPath;
#            // Add the potential k-shortest path to the heap.
#            B.append(totalPath);
           
#            // Add back the edges and nodes that were removed from the graph.
#            restore edges to Graph;
#            restore nodes in rootPath to Graph;
                   
#        if B is empty:
#            // This handles the case of there being no spur paths, or no spur paths left.
#            // This could happen if the spur paths have already been exhausted (added to A), 
#            // or there are no spur paths at all - such as when both the source and sink vertices 
#            // lie along a "dead end".
#            break;
#        // Sort the potential k-shortest paths by cost.
#        B.sort();
#        // Add the lowest cost path becomes the k-shortest path.
#        A[k] = B[0];
#        B.pop();
   
#    return A;

    def yen_KSP(self, start: Hashable, end: Hashable, K: int = 5) -> List[List[Hashable]]:
        """
        https://en.wikipedia.org/wiki/Yen%27s_algorithm
        """
        A = [self.uniform_cost_search(start, end)]
        B = []

        for k in range(1, K):
            for i in range(len(A[k-1])-2):
                spur, root_path = A[k-1][i], A[k-1][:i+1]
                removed_edges = []
                removed_nodes = []
                for path in A:
                    if root_path == path[:i+1]:
                        source = path[i]
                        target = path[i+1]
                        if self.edges[source].pop(target):
                            removed_edges.append((source, target))

                for node in root_path:
                    if node == spur:
                        continue
                    self.nodes.remove(node)

                spur_path = self.uniform_cost_search(spur, end)

                full_path = root_path.extend(spur_path)
                B.append(full_path)

                for node in removed_nodes:
                    self.add(node)
                
                for edge in removed_edges:
                    self.edge(edge[0], edge[1])

            if not B:
                break

            B.sort(key=lambda x: len(x))
            A[k] = B[0]

            B.pop()

        return A


