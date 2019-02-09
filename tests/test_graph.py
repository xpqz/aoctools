from typing import Iterable, List, Set, Tuple
from graph import Graph, taxidistance, CostedPath

Pos = Tuple[int, int]

DATA = [
    list(".....................XX......."),
    list(".....................XX......."),
    list(".....................XX......."),
    list("...XX................XX......."),
    list("...XX................XX......."),
    list("...XX........XX......XXT......"),
    list("...XX........XX......XXXXX...."),
    list("...XX........XX......XXXXX...."),
    list("..SXX........XX..............."),
    list("...XX........XX..............."),
    list("...XX........XX..............."),
    list("...XX........XX..............."),
    list(".............XX..............."),
    list(".............XX..............."),
    list(".............XX...............")
]

def from_rows(graph: Graph[Pos], data: List[List[str]]) -> Tuple[Pos, Pos]:
    """
    Make a grid-like graph based on a simple string notation.
    """
    start: Pos = (0, 0)
    end: Pos = (0, 0)

    nodes: Set[Pos] = set()

    for y, row in enumerate(data):
        for x, ch in enumerate(row):
            pos = (x, y)
            if ch in {".", "S", "T"}:
                nodes.add(pos)
            if ch == "S":
                start = pos
            elif ch == "T":
                end = pos

    for (x, y) in nodes:
        for n in {(x, y-1), (x-1, y), (x, y+1), (x+1, y)}:
            if n in nodes:
                graph.edge((x, y), n)
                graph.edge(n, (x, y))

    return start, end

def render_path(graph: Graph[Pos], path: Iterable[Pos], xsize: int, ysize: int) -> List[str]:
    """
    Render a path on a grid-like graph based on a simple string notation.
    """
    data: List[str] = []
    for y in range(ysize):
        row = ""
        for x in range(xsize):
            pos = (x, y)
            if pos in path:
                row += "*"
            elif pos in graph.edges:
                row += "."
            else:
                row += "X"
        data.append(row)

    return data

def test_sample_data():
    g = Graph[str]()
    start, end = from_rows(g, DATA)

    path = g.uniform_cost_search(start, end)
    path2 = g.a_star_search(start, end)
    path3 = g.breadth_first_search(start, end)

    output = render_path(g, path.path, len(DATA[0]), len(DATA))
    output2 = render_path(g, path2.path, len(DATA[0]), len(DATA))
    output3 = render_path(g, path3, len(DATA[0]), len(DATA))

    print()
    for row in output:
        print(row)

    print()
    for row in output2:
        print(row)

    print()
    for row in output3:
        print(row)

def test_uniform_cost_search():
    g = Graph[str]()

    g.edge("C", "D", 3)
    g.edge("D", "F", 4)
    g.edge("C", "E", 2)
    g.edge("E", "D", 1)
    g.edge("E", "F", 2)
    g.edge("E", "G", 3)
    g.edge("F", "G", 2)
    g.edge("G", "H", 2)
    g.edge("F", "H", 1)

    assert g.uniform_cost_search("C", "H") == CostedPath(5, ["C", "E", "F", "H"])


def test_yen():
    g = Graph[str]()

    g.edge("C", "D", 3)
    g.edge("D", "F", 4)
    g.edge("C", "E", 2)
    g.edge("E", "D", 1)
    g.edge("E", "F", 2)
    g.edge("E", "G", 3)
    g.edge("F", "G", 2)
    g.edge("G", "H", 2)
    g.edge("F", "H", 1)

    paths = g.k_shortest_paths("C", "H", 3)

    assert paths[0] == CostedPath(5, ["C", "E", "F", "H"])
    assert paths[1] == CostedPath(7, ["C", "E", "G", "H"])
    assert paths[2] == CostedPath(8, ["C", "D", "F", "H"])
