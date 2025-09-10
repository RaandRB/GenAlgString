from fitness_functions import *
from functions import *
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import matplotlib.pyplot as plt
import seaborn as sns

r = 2

def wolfram_stability(model, M):
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

def _stability(self):
    r"""Determines if V is slope stable.
    Uses external function.
    
    Returns
    -------
    bool
        True if satisfied
    """
    stable = False
    # need to solve numerically

    #stable = scipy_stability(self.V, self.M)
    #stable = nlopt_stability(self.V, self.M)

    stable = wolfram_stability(self, M)
    
    
    return stable

def _three_fermions(self):
    r"""Determines if the model has exactly three fermion
    generations.

    Returns
    -------
    bool
        True if satisfied
    """
    for entry in self:
        h = np.array(M.line_co(entry)).astype(np.int16)
        if h[2] > 0 or h[0] > 0 or h[3] > 0 or h[1]%r != 0:
            # we found antifamilies/stability problems/no equivariant
            return False
    return True

def _higgs_doublet(self):
    r"""Determines if at least one Higgs doublet exist.
    
    Returns
    -------
    bool
        True if satisfied
    """
    h = np.array([0 for i in range(4)], dtype=np.int16)

    for e1, e2 in itertools.combinations(self, 2):
        l = np.add(list(e1), list(e2))
        h = np.add(h, np.array(M.line_co(l)).astype(np.int16))

    if h[2] == 0:
        return False
    else:
        return True

def _index_triplet(self):
    r"""Determines if there are no Higgs triplets.
    
    Returns
    -------
    bool
        True if satisfied.
    """
    # check that the index of two L_a x L_b is always smaller than zero.
    # Necessary condition to project out all Higgs triplets
    for e1, e2 in itertools.combinations(self, 2):
        l = np.add(list(e1), list(e2))
        # we round and convert to int otherwise floating point issues
        index = np.round(M.line_co_euler(l)).astype(np.int16)
        if index > 0:
            return False

    return True

def _sun(self):
    r"""Determines if c_1(V) = 0.
    Reward is variable and punishes for a far away distance from
    a vanishing first Chern class.
    
    Returns
    -------
    tuple(bool, float)
        (satisfied, reward)
    """
    x = np.sum(self, axis=0)
    if np.all(x == [0 for i in range(M.len)]):
        return True
    else:
        return False

def _bianchi(self):
    r"""Checks anomaly cancellation, i.e. c_2(X) - c_2(V) >= 0.
    Also checks necessary stability i.e. c2(V) must not be negative in all entries.

    Returns
    -------
    bool
        True if satisfied
    """
    c2 = -1/2*np.einsum('rst,st -> r', M.triple, np.einsum('is,it->st', self, self))
    if np.all(np.subtract(M.second_chern(), c2) >= 0) and not np.array_equal(np.sign(c2), np.zeros(len(c2)-1)):
        return True
    else:
        return False

def _index(self):
    r"""Determines if the index tells us that there are three 
    Fermion generations.
    
    Rewards depend on how many index constraint are satisfied
    1) Some: satisfied/50*self.reward_index
    2) All: 0.1*self.reward_index
    3) full V: self.reward_index

    Returns
    -------
    tuple(bool, float)
        (satisfied, reward)
    """
    satisfied = 5
    index = np.zeros(5)
    for i in range(5):
        #round and convert to int because of floating point issues.
        
        index[i] = np.round(M.line_co_euler(self[i])).astype(np.int16)
        # check if index in range and divisible by rank for equivariant structure
        if index[i] > 0 or index[i] < (-3)*r or index[i]%r != 0:
            satisfied -= 1
    
    if satisfied != 5:
        return False

    # sum of index has to be == -3*self.r
    findex = np.sum(index)
    # index has to be <= 0
    if findex != (-3)*r:
        return False
    else:
        return True


#M = CICY([[1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1]])
#M = CICY([[1,2], [1,2], [1,2], [1,2]])
#M = CICY([[2, 1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 1, 0], 
#    [1, 0, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 1, 0, 0], 
#    [1, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 1], 
#    [2, 0, 0, 0, 1, 1, 1, 0, 0], [2, 0, 0, 0, 0, 0, 1, 1, 1]])
#M = CICY([[1, 1, 1, 0, 0, 0, 0, 0, 0], [2, 0, 1, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 2, 0, 0, 0], 
#        [1, 0, 0, 0, 1, 1, 0, 0, 0], [2, 1, 0, 0, 0, 0, 1, 0, 1], [3, 0, 0, 0, 1, 1, 0, 1, 1]])
M = CICY([[1,0,1,1], [1,0,1,1], [1,1,1,0], [1,1,1,0], [1,1,0,1], [1,1,0,1]])
M = CICY([[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
          [2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])


x = [[0,1,2,0,-1,-2,-3,2], [0,1,2,0,-1,-2,-3,2], [0,1,2,0,-1,-2,-3,2]]

y = [[1,0,2,0,-1,-2,-3,2], [1,0,2,0,-1,-2,-3,2], [1,0,2,0,-1,-2,-3,2]] 

t1 = solution_to_graph829(x)

t2 = solution_to_graph829(y)

sols = np.load("solutions.npy")

print(len(sols))
count = 0
for s in sols:
    s = np.array(s)

    x = solution_to_graph(s)

    emb = _sun(s)
    bian = _bianchi(s)
    gen = _three_fermions(s)
    index2 = _index_triplet(s)
    index = _index(s)
    higgs = _higgs_doublet(s)

    if emb and bian and gen and index2 and index and higgs:
        count += 1
print(count)

