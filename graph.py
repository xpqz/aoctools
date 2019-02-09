"""
Simple graph and various search and path algorithms.

A lot of this comes from the excellent article

https://www.redblobgames.com/pathfinding/a-star/introduction.html

"""
import heapq
from collections import defaultdict, deque
from typing import (
    cast, overload, Callable, Dict, Generic,
    List, Optional, Tuple, Union, TypeVar
)
from dataclasses import dataclass

T = TypeVar("T")
Previous = Dict[T, Optional[T]]
CumulativeCost = Dict[T, int]
Nodes = List[T]

class GraphException(Exception):
    pass

@dataclass(eq=True, frozen=True)
class CostedPath(Generic[T]):
    cost: int
    path: Nodes

class PriorityQueue(Generic[T]):
    def __init__(self):
        self.elements: List[Tuple[int, T]] = []

    def empty(self) -> bool:
        return len(self.elements) == 0

    def put(self, item: T, priority: int) -> None:
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        item = heapq.heappop(self.elements)
        return item[1]

    def contains(self, item: T) -> bool:
        return item in self.elements

def taxidistance(target: Tuple[int, int], neighbour: Tuple[int, int]) -> int:
    """
    Common A* heuristic -- the "taxidistance" also known as "Manhattan distance".
    """
    return abs(neighbour[0]-target[0]) + abs(neighbour[1]-target[1])

def nocost(_target: T, _neighbour: T) -> int:
    return 0

class Graph(Generic[T]):
    """
    Simple graph representation. Nodes must be hashable.
    """
    def __init__(self):
        self.edges = defaultdict(dict)

    def edge(self, start: T, end: T, cost: int = 1) -> None:
        self.edges[start][end] = cost

    def unwind_path(self, came_from: Previous, start: T, end: T) -> Nodes:
        current = end
        path: Nodes = []

        while current != start:
            path.append(current)
            try:
                current = cast(T, came_from[current])
            except KeyError:
                raise GraphException("Invalid path")

        path.append(start)
        path.reverse()

        return path


    @overload
    def breadth_first_search(self, start: T) -> Previous: ...

    @overload
    def breadth_first_search(self, start: T, end: T) -> Nodes: ...

    def breadth_first_search(self, start: T, end: T = None) -> Union[Previous, Nodes]:
        """
        Breadth-first search with potential early exit. Edge cost not accounted for.
        If no `end` node given, the returned dict holds the shortest paths from
        the start to every other point in the graph.
        """
        frontier = deque([start])
        came_from: Previous = {start: None}

        while frontier:
            current = frontier.popleft()

            if current == end:
                break

            if current in self.edges:
                for neighbour in self.edges[current].keys():
                    if neighbour not in came_from:
                        frontier.append(neighbour)
                        came_from[neighbour] = current

        if end is None:
            return came_from

        return self.unwind_path(came_from, start, end)

    def dijkstra(self, start: T, end: T = None) -> Tuple[CumulativeCost, Previous]:
        """
        Dijkstra's uniform cost search finds the lowest-cost path from start to end. Returns the
        raw cost and path sequence dicts for further processing.
        """
        frontier = PriorityQueue[T]()
        frontier.put(start, 0)
        came_from: Previous = {start: None}
        cost_so_far: CumulativeCost = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if current == end:
                break

            if current in self.edges:
                for (neighbour, cost) in self.edges[current].items():
                    new_cost = cost_so_far[current] + cost
                    if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                        cost_so_far[neighbour] = new_cost
                        priority = new_cost
                        frontier.put(neighbour, priority)
                        came_from[neighbour] = current

        return cost_so_far, came_from

    def uniform_cost_search(self, start: T, end: T) -> Optional[CostedPath[T]]:
        """
        Dijkstra's with cost and unwound path for the start->end
        TODO: check that it works for no path found.
        """
        cost_so_far, came_from = self.dijkstra(start, end)

        cost = cost_so_far.get(end, None)
        if cost is None:
            return None

        return CostedPath(cost, self.unwind_path(came_from, start, end))

    def a_star_search(
            self,
            start: T,
            end: T,
            heuristic: Callable[[T, T], int] = nocost
    ) -> CostedPath[T]:
        """
        A* search. The heuristic should be a function that gives a "guess"
        of how well travelling from a to b takes you towards the target.
        This could for example be the "manhattan distance" if the graph is
        looking like a grid:

        def taxidistance(target, neighbour):
            return abs(neighbour[0]-target[0]) + abs(neighbour[1]-target[1])

        As long as the given heuristic does not over-estimate the cost,
        A* converges on the lowest-cost path whilst exploring fewer nodes
        than UCS (Dijkstra) and other graph search algorithms.

        With no heuristic cost, A* is equivalent to Dijkstra
        """
        frontier = PriorityQueue[T]()
        frontier.put(start, 0)
        came_from: Previous = {start: None}
        cost_so_far: CumulativeCost = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if current == end:
                break

            if current in self.edges:
                for (neighbour, cost) in self.edges[current].items():
                    new_cost = cost_so_far[current] + cost
                    if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                        cost_so_far[neighbour] = new_cost
                        priority = new_cost + heuristic(end, neighbour)
                        frontier.put(neighbour, priority)
                        came_from[neighbour] = current

        return CostedPath(cost_so_far[end], self.unwind_path(came_from, start, end))

    def k_shortest_paths(self, source: T, sink: T, K: int = 5) -> List[CostedPath[T]]:
        """
        This is Yen's k-sp algorithm, as described on Wikipedia:

        https://en.wikipedia.org/wiki/Yen%27s_algorithm
        """
        costs, came_from = self.dijkstra(source)  # All paths & costs

        A: List[CostedPath[T]] = [
            CostedPath(costs[sink], self.unwind_path(came_from, source, sink))
        ]
        B: List[Tuple[int, Nodes]] = []  # heapq

        for _k in range(1, K):
            # The spur node ranges from the first node to the neighbour to
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

                    # Add the potential k-shortest path to the heap
                    item = (full_cost, full_path)
                    if item not in B:
                        heapq.heappush(B, item)

                    # Add back the edges and nodes that were removed from the graph.
                    for (a, b, cost) in removed_edges:
                        self.edge(a, b, cost)

            if not B:
                # This handles the case of there being no spur paths, or no spur paths left.
                # This could happen if the spur paths have already been exhausted (added to A),
                # or there are no spur paths at all - such as when both the source and sink vertices
                # lie along a "dead end".
                break

            # The lowest cost path becomes the kth-shortest path.
            A.append(CostedPath(*heapq.heappop(B)))

        return A
