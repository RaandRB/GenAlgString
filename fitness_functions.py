from deap import base, creator, tools
from pyCICY import CICY
import time
import itertools
import numpy as np
from functions import *
import pickle
import collections



#C1
def embedding_check(M, V):
    #M: CICY class from pyCICY
    #V: Vector bundle, numpy list of 5 line bundles, one line bundle is a list of k integers, where k = h^(1,1)(M).
    
    #c_1(V) is already fixed to be 0, so this checks if any proper subsets have vanishing c_1.
    t1 = time.time()
    emb_sum = 0
    for s in range(1,len(V)):

        idx = np.array(list(itertools.combinations(range(len(V)), s)))

        sub_sums = V[idx].sum(axis=1)

        zero_vecs = np.sum(~np.any(sub_sums, axis=1))

        emb_sum -= zero_vecs

        #for set in subsets:
        #    summed_set = np.sum(set, axis=0)
        #    if not summed_set.any():
        #        emb_sum -= 1

    t2 = time.time()
    #print(f"Embedding check (c1) took {t2-t1:.8f} seconds to complete.")
    return emb_sum*10




#C2

def anomaly_check(M, V):
    #M: CICY class from pyCICY
    #V: Vector bundle, numpy list of 5 line bundles, one line bundle is a list of k integers, where k = h^(1,1)(M).

    anom_sum = 0

    c2_TM = chern_second(M)

    triple_intersect_vec = M.triple_intersection()

    inner_contraction = np.einsum("aj,ak->jk", V, V)

    tot_contraction = np.einsum("ijk,jk->i", np.array(triple_intersect_vec), inner_contraction)

    c2_V = -(1/2)*tot_contraction

    anom_sum = np.sum(np.minimum(0, c2_TM-c2_V))

    return anom_sum*1

def anomaly_check_strict(M, V):
    #M: CICY class from pyCICY
    #V: Vector bundle, numpy list of 5 line bundles, one line bundle is a list of k integers, where k = h^(1,1)(M).

    anom_sum = 0

    c2_TM = chern_second(M)

    triple_intersect_vec = M.triple_intersection()

    c2_V = np.zeros(np.shape(V)[1])

    for i in range(np.shape(V)[1]):
        inner_contraction = np.einsum("aj,ak->jk", V, V)
        tot_contraction = np.einsum("jk,jk", np.array(triple_intersect_vec[i]), inner_contraction)
        c2_V[i] = tot_contraction
        anom_sum += abs(c2_TM[i] - c2_V[i])

    return -anom_sum


#C3


def slope_check_real(M, V):


    is_stable = wolfram_stability(M, V)

    slope_sum = 0

    if not is_stable:
        slope_sum -= 15
 

    return slope_sum


def slope_check_article(M , V):
    d_vec = np.array(M.triple_intersection())
    M_mat = np.einsum("ijk, ai-> ajk", d_vec, V)
    with open("out.pkl", "rb") as f:
        arr = pickle.load(f)
    all_M = np.tensordot(arr, M_mat, axes=([1], [0]))

    all_pos = np.sum(np.all(all_M >= 0, axis=(1,2)) & np.any(all_M != 0, axis=(1,2)))
    all_neg = np.sum(np.all(all_M <= 0, axis=(1,2)) & np.any(all_M != 0, axis=(1,2)))
    


    return -(all_pos + all_neg)

def slope_check_article_vectorized(M, V):
    #Now V is shape (nV, 5, h)
    d_vec = np.array(M.triple_intersection())
    M_mat = np.einsum("ijk, nai-> najk", d_vec, V)
    with open("out.pkl", "rb") as f:
        arr = pickle.load(f)
    all_M = np.tensordot(arr, M_mat, axes=([1], [1]))

    all_pos = np.sum(np.all(all_M >= 0, axis=(2,3)) & np.any(all_M != 0, axis=(2,3)), axis=0)
    all_neg = np.sum(np.all(all_M <= 0, axis=(2,3)) & np.any(all_M != 0, axis=(2,3)), axis=0)
    


    return -(all_pos + all_neg)/10

#C4

def cohom_check(M, V, free_size):
    line_cohoms = np.array([M.line_co(L, SpaSM=True) for L in V])
    vec_cohoms = line_cohoms.sum(axis=0)
    h1 = np.sum(vec_cohoms[1])
    h2 = np.sum(vec_cohoms[2])

    #wedge2_cohoms = np.array([M.line_co(Li + Lj, SpaSM=True) for Li, Lj in itertools.combinations(V, 2)])
    #wedge2_cohoms = wedge2_cohoms.sum(axis=0)

    #h1_wedge2V, h2_wedge2V = wedge2_cohoms[1], wedge2_cohoms[2]

    diff_h1 = abs(h1-3*free_size)
    diff_h2 = h2
    #diff_h1w = abs(h1_wedge2V-(3*free_size + h2_wedge2V))

    result = diff_h1 + diff_h2
    #print(result)

    return -result

def cohom_check_total(M, V, free_size):
    V = [np.array(v) for v in V]
    line_cohoms = np.array([M.line_co(L, SpaSM=True) for L in V])
    vec_cohoms = line_cohoms.sum(axis=0)
    h1 = np.sum(vec_cohoms[1])
    h2 = np.sum(vec_cohoms[2])

    wedge2_cohoms = np.array([M.line_co(Li + Lj, SpaSM=True) for Li, Lj in itertools.combinations(V, 2)])
    wedge2_cohoms = wedge2_cohoms.sum(axis=0)

    h1_wedge2V, h2_wedge2V = wedge2_cohoms[1], wedge2_cohoms[2]

    diff_h1 = abs(h1-3*free_size)
    diff_h2 = h2
    diff_h1w = abs(h1_wedge2V-(3*free_size + h2_wedge2V))

    if h2_wedge2V == 0:
        diff_h2w = 10
    else:
        diff_h2w = 0

    result = diff_h1 + diff_h2 + diff_h1w + diff_h2w

    return -result

def cohom_euler_check(M, V, free_size):
    
    euler_V = [np.rint(M.line_co_euler(L)) for L in V]

    result = 0

    for e in euler_V:
        if e > 0:
            result += 10
    
    tot_euler = np.sum(euler_V)

    result += np.abs(tot_euler + 3*free_size)

    for e1, e2 in itertools.combinations(V, 2):
        l = np.add(list(e1), list(e2))
        # we round and convert to int otherwise floating point issues
        index = np.round(M.line_co_euler(l)).astype(np.int16)
        #if index > 0:
            #result += 25


    return -result*10


#C5

def equiv_check(M, V, free_size):
    unique_L, counts = np.unique(V, return_counts=True, axis=0)

    occ_L = list(zip(unique_L, counts))

    equiv_result = 0
    for L, count in occ_L:
        euler_L = np.rint(M.line_co_euler(L))*count
        equiv_result -= abs(euler_L%free_size)
        #if equiv_result != 0:
        #    print("test")

    return equiv_result*10


def gamma_orbit(L, gamma):
    orbit = []
    acted_L = tuple(np.matmul(L, gamma))

    while acted_L not in orbit:
        orbit.append(acted_L)
        acted_L = tuple(np.matmul(np.array(acted_L), gamma))
    return orbit


def invariant_partition_or_penalty(V, Gamma, penalty=-100):
    V_counter = collections.Counter(map(tuple, V))
    seen = set()
    partitions = []

    penalty = 0
    passed = True
    
    for v in list(V_counter):
        if v not in seen:
            orbit = gamma_orbit(v, Gamma)
            seen.update(orbit)
            
            # If orbit element missing, fail
            if any(o not in V_counter for o in orbit):
                penalty -= 30
                passed = False
                continue

            
            # All multiplicities in the orbit must be equal
            counts = [V_counter[o] for o in orbit]
            if len(set(counts)) != 1:
                passed = False
                penalty -= 15
            
            # Build a block with those multiplicities
            block = []
            for o in orbit:
                block.extend([o] * V_counter[o])
            partitions.append(block)

    if not passed:
        return penalty, False
    
    return partitions, True

def equiv_nontriv829(M, V, free_size):
    Gamma = np.array([
            [0,1,0,0,0,0,0,0], 
            [0,0,1,0,0,0,0,0],  
            [1,0,0,0,0,0,0,0],  
            [0,0,0,0,1,0,0,0],  
            [0,0,0,0,0,1,0,0],  
            [0,0,0,1,0,0,0,0],  
            [0,0,0,0,0,0,1,0],  
            [0,0,0,0,0,0,0,1]])
    partition, passed = invariant_partition_or_penalty(V, Gamma)
    score = 0
    
    if not passed:
        return partition
    
    for block in partition:
        block_euler = 0
        for L in block:
            block_euler += np.rint(M.line_co_euler(np.array(L)))

        if block_euler%free_size != 0:
            score -= 20

    return score*50

def equiv_nontriv4071(M, V, free_size):
    Gamma = [[1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]]
    partition, passed = invariant_partition_or_penalty(V, Gamma)
    score = 0
    
    if not passed:
        return partition
    
    for block in partition:
        block_euler = 0
        for L in block:
            block_euler += np.rint(M.line_co_euler(np.array(L)))

        if block_euler%free_size != 0:
            score -= 20

    return score*50


#Genalg functions


def tot_bundle(ind):
    last_bundle = [-sum(col) for col in zip(*ind)]
    tot_bund = [row[:] for row in ind]
    tot_bund.append(last_bundle)
    return tot_bund

def evaluate(individual, M, free_size):

    ind = np.array(tot_bundle(individual))

    emb_test = embedding_check(M, ind)
    anom_test = anomaly_check(M, ind)
    slope_test = slope_check_article(M, ind)
    cohom_test = cohom_euler_check(M, ind, free_size=free_size)
    equiv_test = equiv_check(M, ind, free_size=free_size)

    score = emb_test + anom_test + slope_test + cohom_test + equiv_test
    passed = sum([emb_test==0, anom_test==0, slope_test==0, cohom_test==0, equiv_test==0])

    return score, passed



def evaluate_nontriv829(individual, M, free_size):

    ind = np.array(tot_bundle(individual))

    emb_test = embedding_check(M, ind)
    anom_test = anomaly_check(M, ind)
    slope_test = slope_check_article(M, ind)
    cohom_test = cohom_euler_check(M, ind, free_size=free_size)
    equiv_test = equiv_nontriv829(M, ind, free_size=free_size)

    score = emb_test + anom_test + slope_test + cohom_test + equiv_test
    passed = sum([emb_test==0, anom_test==0, slope_test==0, cohom_test==0, equiv_test==0])

    return score, passed

def evaluate_nontriv4071(individual, M, free_size):

    ind = np.array(tot_bundle(individual))

    emb_test = embedding_check(M, ind)
    anom_test = anomaly_check(M, ind)
    slope_test = slope_check_article(M, ind)
    cohom_test = cohom_euler_check(M, ind, free_size=free_size)
    equiv_test = equiv_nontriv4071(M, ind, free_size=free_size)

    score = emb_test + anom_test + slope_test + cohom_test + equiv_test
    passed = sum([emb_test==0, anom_test==0, slope_test==0, cohom_test==0, equiv_test==0])

    return score, passed

#Nested mate
def mate_nested(ind1, ind2):

    for row1, row2 in zip(ind1, ind2):
        tools.cxTwoPoint(row1, row2)
    return ind1, ind2

def mutate_nested(ind, low, up, indpb):
    for row in ind:
        tools.mutUniformInt(row, low=low, up=up, indpb=indpb)
    return ind



M_7862 = CICY([[1,2], [1,2], [1,2], [1,2]])

M_7447 = CICY([[1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1]])

M_5302 = CICY([[1,0,1,1], [1,0,1,1], [1,1,1,0], [1,1,1,0], [1,1,0,1], [1,1,0,1]])

M_4071 = CICY([[1, 1, 1, 0, 0, 0, 0, 0, 0], [2, 0, 1, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 2, 0, 0, 0], 
[1, 0, 0, 0, 1, 1, 0, 0, 0], [2, 1, 0, 0, 0, 0, 1, 0, 1], [3, 0, 0, 0, 1, 1, 0, 1, 1]])

M_829 = CICY([[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])

def choose_eval_cert(M):

    
    if M == M_829:

        return solution_to_graph829, evaluate_nontriv829
    
    if M == M_4071:
        
        return solution_to_graphcert4071, evaluate_nontriv4071

    
    return solution_to_graphcert, evaluate
