from deap import base, creator, tools
from scoop import futures
import random
from pyCICY import CICY
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import z3
from bitarray import bitarray
import networkx as nx
import pynauty
from collections import defaultdict
import cvxpy as cp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

def sympy_to_z3(expr):
    z3_vars = {s: z3.Real(str(s)) for s in expr.free_symbols}
    if expr.is_Number:
        return z3.RealVal(float(expr))
    if expr.is_Symbol:
        return z3_vars[expr]
    if expr.is_Add:
        return sum(sympy_to_z3(a) for a in expr.args)
    if expr is sp.true:
        return z3.BoolVal(True)
    if expr is sp.false:
        return z3.BoolVal(False)
    if expr.is_Mul:
        prod = sympy_to_z3(expr.args[0])
        for a in expr.args[1:]:
            prod *= sympy_to_z3(a)
        return prod
    if expr.is_Pow:
        base, exp = expr.args
        return sympy_to_z3(base) ** sympy_to_z3(exp)
    if expr.is_Relational:
        lhs, rhs = expr.lhs, expr.rhs
        op = expr.rel_op
        if op == '==':
            return sympy_to_z3(lhs) == sympy_to_z3(rhs)
        elif op == '!=':
            return sympy_to_z3(lhs) != sympy_to_z3(rhs)
        elif op == '<':
            return sympy_to_z3(lhs) < sympy_to_z3(rhs)
        elif op == '<=':
            return sympy_to_z3(lhs) <= sympy_to_z3(rhs)
        elif op == '>':
            return sympy_to_z3(lhs) > sympy_to_z3(rhs)
        elif op == '>=':
            return sympy_to_z3(lhs) >= sympy_to_z3(rhs)
    raise NotImplementedError(f"Unsupported sympy construct: {expr}")


def sympy_to_z32(sympy_var_list, sympy_exp):
    'convert a sympy expression to a z3 expression. This returns (z3_vars, z3_expression)'

    z3_vars = []
    z3_var_map = {}

    for var in sympy_var_list:
        name = var.name
        z3_var = z3.Real(name)
        z3_var_map[name] = z3_var
        z3_vars.append(z3_var)

    result_exp = _sympy_to_z3_rec(z3_var_map, sympy_exp)

    return z3_vars, result_exp

def _sympy_to_z3_rec(var_map, e):
    'recursive call for sympy_to_z3()'

    rv = None

    if not isinstance(e, sp.Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, sp.Symbol):
        rv = var_map.get(e.name)

        if rv == None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

    elif isinstance(e, sp.Number):
        rv = float(e)
    elif isinstance(e, sp.Mul):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv *= _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, sp.Add):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv += _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, sp.Pow):
        term = _sympy_to_z3_rec(var_map, e.args[0])
        exponent = _sympy_to_z3_rec(var_map, e.args[1])

        if exponent == 0.5:
            # sqrt
            rv = z3.Sqrt(term)
        else:
            rv = term**exponent

    if rv == None:
        raise RuntimeError("Type '" + str(type(e)) + "' is not yet implemented for convertion to a z3 expresion. " + \
                            "Subexpression was '" + str(e) + "'.")

    return rv


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


def solution_to_graph2(solution, low, up):
    #Note that we assume here that the integers in solution are not less than -18, modify value_to_color if this changes
    num_vecs = len(solution)
    n = len(solution[0])
    num_entries = num_vecs * n
    N = num_vecs + n + num_entries

    g = pynauty.Graph(number_of_vertices=N, directed=False)


    def value_to_color(value):
        return value + 20

    # Colors: vectors = 0, positions = 1, entries = mapped by value_to_color
    colors = [0]*num_vecs + [1]*n
    entry_colors = [value_to_color(solution[i][j]) for i in range(num_vecs) for j in range(n)]
    colors += entry_colors

    color_classes = {}
    for v, c in enumerate(colors):
        color_classes.setdefault(c, []).append(v)
    partition = [set(verts) for verts in color_classes.values()]
    g.set_vertex_coloring(partition)

    # Add edges
    entry_base = num_vecs + n
    for i in range(num_vecs):
        for j in range(n):
            entry_idx = entry_base + i * n + j
            g.connect_vertex(i, [entry_idx])
            g.connect_vertex(num_vecs + j, [entry_idx])

    return g


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

def solution_to_graph829(solution, low=-18, up=18):
    """
    Turn a solution (matrix of shape (num_vecs, n)) into a colored graph
    where integers in [low, up] are explicitly represented as vertices.
    """

    num_vecs = len(solution)
    n = len(solution[0])
    num_entries = num_vecs * n
    num_distinct_kähler = 8 # 1 1 1 2 2 2 2 2
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
    # vecs = 0, positions = 1, entries = 2, values = 3,
    colors = [0] * num_vecs + [1] * n + [2] * num_entries
    for i in range(8):
        color += [kähler_offset + i]
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

            g.connect_vertex(entry_idx, [kähler_offset + j])

            value = int(solution[i][j])
            value_idx = value_to_color(value)
            g.connect_vertex(entry_idx, [value_idx])

    return g
