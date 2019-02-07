from graph import Graph, taxidistance, CostedPath

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



def test_sample_data():
    g = Graph()
    start, end = g.from_rows(DATA)

    path = g.uniform_cost_search(start, end)
    path2 = g.a_star_search(start, end)
    path3 = g.breadth_first_search(start, end)

    output = g.render_path(path, len(DATA[0]), len(DATA))
    output2 = g.render_path(path2, len(DATA[0]), len(DATA))
    output3 = g.render_path(path3, len(DATA[0]), len(DATA))
    
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
    g = Graph()

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
    g = Graph()

    g.edge("C", "D", 3)
    g.edge("D", "F", 4)
    g.edge("C", "E", 2)
    g.edge("E", "D", 1)
    g.edge("E", "F", 2)
    g.edge("E", "G", 3)
    g.edge("F", "G", 2)
    g.edge("G", "H", 2)
    g.edge("F", "H", 1)

    paths = g.yen_KSP("C", "H", 3)

    assert paths[0] == CostedPath(5, ["C", "E", "F", "H"])
    assert paths[1] == CostedPath(7, ["C", "E", "G", "H"])
    assert paths[2] == CostedPath(8, ["C", "D", "F", "H"])
