from graph import Graph, taxidistance

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


    for p in g.yen_KSP(start, end):
        o = g.render_path(p, len(DATA[0]), len(DATA))
        print()
        for row in o:
            print(row)

    assert len(path) == len(path2)
