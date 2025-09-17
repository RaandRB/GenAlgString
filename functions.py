from pyCICY import CICY
import numpy as np
import matplotlib.pyplot as plt
import pynauty
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr


def chern_second(M):
    Msub = M.M[:, 1:M.K+1]

    dot_matrix = Msub @ Msub.T
    delta = np.diag(M.M[:, 0] + 1)

    c2 = (dot_matrix - delta) / 2.0
    chern = np.einsum('rst,st -> r', M.triple, c2)
    return chern



def wolfram_stability(M, model):
    r"""Use the wolfram Noptimize to numerically compute
    if the given model is stable somewhere in the Kähler cone

    Parameters
    ----------
    model : np.array[5, M.len]
        sum of line bundles
    M : pyCICY.CICY
        CICY object

    Returns
    -------
    bool
        True if stable else False
    """
    session = WolframLanguageSession("/usr/local/bin/WolframKernel")#kernel_loglevel=logging.ERROR
    line_slope = []
    for line in model:
        tmp = M.line_slope()
        for i in range(M.len):
            tmp = tmp.subs({'m'+str(i): line[i]})
        line_slope += [tmp]
    str_slope = str(line_slope).replace('**', '^').replace('[','{').replace(']', '}')
    str_cone = str(['t'+str(i)+'> 1' for i in range(M.len)]).replace('\'', '').replace('[','{').replace(']', '}')
    str_vars = str(['t'+str(i) for i in range(M.len)]).replace('\'', '').replace('[','{').replace(']', '}')
    success = False
    full_string = 'NMinimize[Join[{{Plus @@ (#^2 & /@ {})}}, '.format(str_slope)
    full_string += '{}],'.format(str_cone)
    full_string += '{}, AccuracyGoal -> 20, PrecisionGoal -> 20, WorkingPrecision -> 20]'.format(str_vars)
    optimize = wlexpr(full_string)
    results = session.evaluate(optimize)
    if np.allclose([0.], [results[0].__float__()], atol=0.01):
        success = True
    session.terminate()
    return success

def solution_to_graph(solution, low=-18, up=18):
    """
    Turn a solution (matrix of shape (num_vecs, n)) into a colored graph
    where integers in [low, up] are explicitly represented as vertices.
    """

    num_vecs = len(solution)
    n = len(solution[0])
    num_entries = num_vecs * n
    num_values = up - low + 1

    # vertices = vecs + positions + entries + all possible values
    N = num_vecs + n + num_entries + num_values
    g = pynauty.Graph(number_of_vertices=N, directed=False)

    # Index bookkeeping
    vec_offset = 0
    pos_offset = num_vecs
    entry_offset = pos_offset + n
    value_offset = entry_offset + num_entries

    # Assign colors
    # vecs = 0, positions = 1, entries = 2, values = 3
    colors = [0] * num_vecs + [1] * n + [2] * num_entries
    # Partition
    color_classes = {}
    for v, c in enumerate(colors):
        color_classes.setdefault(c, []).append(v)

    def value_to_color(value):
        return value - low + value_offset

    for k in range(low, up +1):
        color_classes.setdefault(str(k), []).append(value_to_color(k))

    partition = [set(verts) for verts in color_classes.values()]
    g.set_vertex_coloring(partition)

    # Add edges
    for i in range(num_vecs):
        for j in range(n):
            entry_idx = entry_offset + i * n + j
            vec_idx = vec_offset + i
            pos_idx = pos_offset + j
            g.connect_vertex(entry_idx, [vec_idx, pos_idx])

            value = int(solution[i][j])
            value_idx = value_to_color(value)
            g.connect_vertex(entry_idx, [value_idx])

    return g

def solution_to_graphcert(solution, low=-18, up=18):

    return pynauty.certificate(solution_to_graph(solution, low, up))

def solution_to_graph4071(solution, low=-18, up=18):
    """
    Turn a solution (matrix of shape (num_vecs, n)) into a colored graph
    where integers in [low, up] are explicitly represented as vertices.
    """

    num_vecs = len(solution)
    n = len(solution[0])
    num_entries = num_vecs * n
    num_distinct_kähler = 3 # 1 2 1 1 1 2 3
    num_values = up - low + 1

    # vertices = vecs + positions + entries + all possible values
    N = num_vecs + n + num_entries + num_values + num_distinct_kähler
    g = pynauty.Graph(number_of_vertices=N, directed=False)

    # Index bookkeeping
    vec_offset = 0
    pos_offset = num_vecs
    entry_offset = pos_offset + n
    kähler_offset = entry_offset + num_entries
    value_offset = kähler_offset + num_distinct_kähler

    # Assign colors
    # vecs = 0, positions = 1, entries = 2, values = 3, P^1 = 100 P^2 = 200 P^3 = 300
    colors = [0] * num_vecs + [1] * n + [2] * num_entries + [kähler_offset] + [kähler_offset+1] + [kähler_offset+2]
    # Partition
    color_classes = {}
    for v, c in enumerate(colors):
        color_classes.setdefault(c, []).append(v)

    def value_to_color(value):
        return value - low + value_offset

    for k in range(low, up +1):
        color_classes.setdefault(str(k), []).append(value_to_color(k))

    partition = [set(verts) for verts in color_classes.values()]
    g.set_vertex_coloring(partition)

    # Add edges
    for i in range(num_vecs):
        for j in range(n):
            entry_idx = entry_offset + i * n + j
            vec_idx = vec_offset + i
            pos_idx = pos_offset + j
            g.connect_vertex(entry_idx, [vec_idx, pos_idx])

            if j in [0, 2, 3, 4]:
                g.connect_vertex(entry_idx, [kähler_offset])
            elif j in [1, 5]:
                g.connect_vertex(entry_idx, [kähler_offset+1])
            elif j == 6:
                g.connect_vertex(entry_idx, [kähler_offset+2])

            value = int(solution[i][j])
            value_idx = value_to_color(value)
            g.connect_vertex(entry_idx, [value_idx])

    return g

def solution_to_graphcert4071(solution, low=-18, up=18):

    return pynauty.certificate(solution_to_graph4071(solution, low, up))

def solution_to_graph829(solution, low=-18, up=18):
    """
    Turn a solution (matrix of shape (num_vecs, n)) into a colored graph
    where integers in [low, up] are explicitly represented as vertices.
    """

    num_vecs = len(solution)
    n = len(solution[0])
    num_entries = num_vecs * n
    num_values = up - low + 1

    # vertices = vecs + positions + entries + all possible values
    N = num_vecs + n + num_entries + num_values
    g = pynauty.Graph(number_of_vertices=N, directed=False)

    # Index bookkeeping
    vec_offset = 0
    pos_offset = num_vecs
    entry_offset = pos_offset + n
    value_offset = entry_offset + num_entries

    # Assign colors
    # vecs = 0, positions = 1, entries = 2, values = 3, P^1 = 100 P^2 = 200 P^3 = 300
    colors = [0] * num_vecs + [1] * n
    # Partition
    color_classes = {}
    for v, c in enumerate(colors):
        color_classes.setdefault(c, []).append(v)

    for i in range(num_entries):
        color_classes.setdefault("Entry " + str(i), []).append(entry_offset + i)

    def value_to_color(value):
        return value - low + value_offset

    for k in range(low, up +1):
        color_classes.setdefault(str(k), []).append(value_to_color(k))

    partition = [set(verts) for verts in color_classes.values()]
    g.set_vertex_coloring(partition)

    # Add edges
    for i in range(num_vecs):
        for j in range(n):
            entry_idx = entry_offset + i * n + j
            vec_idx = vec_offset + i
            pos_idx = pos_offset + j
            g.connect_vertex(entry_idx, [vec_idx, pos_idx])

            value = int(solution[i][j])
            value_idx = value_to_color(value)
            g.connect_vertex(entry_idx, [value_idx])

    return g

def solution_to_graphcert829(solution, low=-18, up=18):
    Gamma = np.array([
        [0,1,0,0,0,0,0,0],  # J1 -> J2
        [0,0,1,0,0,0,0,0],  # J2 -> J3
        [1,0,0,0,0,0,0,0],  # J3 -> J1
        [0,0,0,0,1,0,0,0],  # J4 -> J5
        [0,0,0,0,0,1,0,0],  # J5 -> J6
        [0,0,0,1,0,0,0,0],  # J6 -> J4
        [0,0,0,0,0,0,1,0],  # J7 fixed
        [0,0,0,0,0,0,0,1],  # J8 fixed
    ])

    sol2 = [np.matmul(s, Gamma) for s in solution]

    sol3 = [np.matmul(s, Gamma) for s in sol2]

    cert1 = pynauty.certificate(solution_to_graph829(solution, low, up))

    cert2 = pynauty.certificate(solution_to_graph829(sol2, low, up))

    cert3 = pynauty.certificate(solution_to_graph829(sol3, low, up))

    cert_vec = sorted([cert1, cert2, cert3])

    return cert_vec[0] + cert_vec[1] + cert_vec[2]



def adaptive_mut(best, indpb):
    upper_limit = 0.6
    lower_limit = 0.3
    rate = 0.04

    if best == 0:
        return max(lower_limit, indpb*(1-rate))
    else:
        return min(upper_limit, indpb*(1+rate)) 



def assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist


