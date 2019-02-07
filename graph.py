"""
Simple graph and various search and path algorithms.

A lot of this comes from the excellent article

https://www.redblobgames.com/pathfinding/a-star/introduction.html

"""
from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import math
from typing import (
    cast, Any, Callable, DefaultDict, Dict, Hashable, 
    Iterable, List, Mapping, Optional, Set, Tuple
)

class GraphException(Exception):
    pass

@dataclass
class CostedPath:
    cost: int
    path: List[Hashable]

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
        self.edges = defaultdict(dict)

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
                return []

        path.append(start)
        path.reverse()

        return path

    def from_rows(self, data: List[List[str]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Primarily for testing -- make a grid-like graph based on a simple string notation.
        """
        start: Tuple[int, int] = (0, 0)
        end: Tuple[int, int] = (0, 0)

        nodes = set()
         
        for y, row in enumerate(data):
            for x, ch in enumerate(row):
                pos = (x, y)
                if ch in {".", "S", "T"}:
                    nodes.add(pos)
                if ch == "S":
                    start = pos
                elif ch == "T":
                    end = pos

        for node in nodes:
            (x, y) = cast(Tuple[int, int], node)
            for n in {(x, y-1), (x-1, y), (x, y+1), (x+1, y)}:
                if n in nodes:
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
                elif pos in self.edges:
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

    def dijkstra(self, start: Hashable, end: Hashable = None) -> Tuple[Dict[Hashable, int], Dict[Hashable, Any]]:
        """
        Dijkstra's uniform cost search finds the lowest-cost path from start to end. Returns the 
        raw cost and path sequence dicts for further processing.
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

        return cost_so_far, came_from

    def uniform_cost_search(self, start: Hashable, end: Hashable) -> Optional[CostedPath]:
        """
        Dijkstra's with cost and unwound path for the start->end
        """
        cost_so_far, came_from = self.dijkstra(start, end)

        cost = cost_so_far.get(end, None)
        if cost is None:
            return None

        return CostedPath(cost, self._unwind_path(came_from, start, end))

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

    def yen_KSP(self, source: Hashable, sink: Hashable, K: int = 5) -> List[CostedPath]:
        """
        https://en.wikipedia.org/wiki/Yen%27s_algorithm
        """
        costs, came_from = self.dijkstra(source)
        
        A = [CostedPath(costs[sink], self._unwind_path(came_from, source, sink))]
        B: List[CostedPath] = []

        for _k in range(1, K):
            # The spur node ranges from the first node to the next to 
            # last node in the previous k-shortest path.
            for i in range(len(A[-1].path)-1):
                # Spur node is retrieved from the previous k-shortest path, k − 1.
                spur = A[-1].path[i]

                # root_path is the sequence of nodes from the source to the spur node 
                # of the previous k-shortest path.
                root_path = A[-1].path[:i+1]

                removed_edges = []
                for p in A:
                    path = p.path
                    if len(path) > i and root_path == path[:i+1]:
                        # Remove the links that are part of the previous shortest paths 
                        # which share the same root path.
                        if path[i] in self.edges:
                            cost = self.edges[path[i]].pop(path[i+1], None)
                            if cost is not None:
                                removed_edges.append((path[i], path[i+1], cost))
               
                # Calculate the spur path from the spur node to the sink.
                costed_spur_path = self.uniform_cost_search(spur, sink)

                # Entire path is made up of the root path and spur path.
                if costed_spur_path is not None:
                    full_path = root_path[:-1] + costed_spur_path.path
                    full_cost = costed_spur_path.cost + costs[spur]

                    # Add the potential k-shortest path to the "heap"
                    item = CostedPath(full_cost, full_path)
                    if item not in B:
                        B.append(item)

                    # Add back the edges and nodes that were removed from the graph.              
                    for (a, b, cost) in removed_edges:
                        self.edge(a, b, cost)

            if not B:
                # This handles the case of there being no spur paths, or no spur paths left.
                # This could happen if the spur paths have already been exhausted (added to A), 
                # or there are no spur paths at all - such as when both the source and sink vertices 
                # lie along a "dead end".
                break

            # Sort the potential k-shortest paths by cost.
            B.sort(key=lambda x: x.cost)

            # The lowest cost path becomes the k-shortest path.
            A.append(B[0])

            B.pop(0)

        return A


