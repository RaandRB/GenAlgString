from deap import base, creator, tools
from scoop import futures
import random
from pyCICY import CICY
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from fitness_functions import *
from functions import *
import pickle
from bitarray import bitarray
import pandas as pd
import seaborn as sns
from line_profiler import LineProfiler
import sklearn as sk
import secrets
import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm


#M = CICY([[1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1]])
#M = CICY([[1,0,1,1], [1,0,1,1], [1,1,1,0], [1,1,1,0], [1,1,0,1], [1,1,0,1]])
#M = CICY([[1,2], [1,2], [1,2], [1,2]])
#M = CICY([[1, 1, 1, 0, 0, 0, 0, 0, 0], [2, 0, 1, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 2, 0, 0, 0], 
#        [1, 0, 0, 0, 1, 1, 0, 0, 0], [2, 1, 0, 0, 0, 0, 1, 0, 1], [3, 0, 0, 0, 1, 1, 0, 1, 1]])
M = CICY([[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
          [2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])



free_size = 2

#How many k's
LINE_SIZE =  M.len

#How many bundles, (5-1)
IND_SIZE = 4

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("vector_bundle", list, fitness=creator.FitnessMax)

low = -2

up = 2

toolbox = base.Toolbox()

toolbox.register("map", map)
toolbox.register("attribute", random.randint, low, up)


toolbox.register("vec_ind", tools.initRepeat, list, toolbox.attribute, n=LINE_SIZE)
toolbox.register("individual", tools.initRepeat, creator.vector_bundle, toolbox.vec_ind, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("mate", mate_nested)

toolbox.register("mutate", mutate_nested, low=low, up=up)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate_nontriv, M=M, free_size=free_size)



stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("best", np.max)

logbook = tools.Logbook()
logbook.header = "gen", "avg", "best"

def adaptive_mut(best, indpb):
    upper_limit = 0.3
    lower_limit = 0.05
    rate = 0.04

    if best == 0:
        return max(lower_limit, indpb*(1-rate))
    else:
        return min(upper_limit, indpb*(1+rate)) 

def main():
    seed = secrets.randbits(32)
    random.seed(seed)
    solution_certs = set()
    solution_vectors = []
    states = []
    nr_sols = []
    t_vec = []
    perc_good = []

    pop = toolbox.population(n=1000)
    CXPB, MUTPB, NGEN = 0.6, 0.2, 1000
    elitism_size = 0
    indpb = 0.1

    
    fitnesses = toolbox.map(toolbox.evaluate, pop)



    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    
    for g in range(NGEN):
        #t1 = time.time()
        offspring = toolbox.select(pop, len(pop) - elitism_size)
        offspring = list(toolbox.map(toolbox.clone, offspring))


        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant, indpb=indpb)
                del mutant.fitness.values


        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)


        
        count_sols = 0

        current_best = None

        t1=time.time()
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if current_best == None or fit[0] > current_best:
                current_best = fit[0]
                
            vec_sol = tot_bundle(ind)
            if all(f == 0 for f in fit):
                count_sols += 1
                canonical_cert = pynauty.certificate(solution_to_graph829(vec_sol, low=-up*4, up=-low*4))
                if canonical_cert not in solution_certs:
                    solution_certs.add(canonical_cert)
                    solution_vectors.append(vec_sol)
            states.append(np.array(vec_sol))
        perc_good.append(count_sols/300)
        t2 = time.time()
        
        record = stats.compile(pop)

        current_best = record["best"]
        #print(current_best)
        indpb = adaptive_mut(current_best, indpb)


        elitism = tools.selBest(pop, elitism_size)

        offspring.extend(map(toolbox.clone, elitism))

        pop[:] = offspring
        #print(f"Generation {g+1}/{NGEN}")
        
        logbook.record(gen=g, **record)
        nr_sols.append(len(solution_certs))

        
        t_vec.append(t2-t1)

        #print(f"Found {len(solutions)} solutions in {t2-t1:.4f} seconds")

    #print(f"States before filtering: {len(solution_certs)}")
    plot = False

    if plot:
        gen = logbook.select("gen")
        fit_avg = logbook.select("avg")
        best = logbook.select("best")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_avg, "b-", label="Average Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        #ax2 = ax1.twinx()
        #line2 = ax2.plot(gen, perc_good, "r-", label="Percentage of pop sol")
        #ax2.set_ylim(0,1)
        #ax2.set_ylabel("%", color="r")
        #for tl in ax2.get_yticklabels():
        #    tl.set_color("r")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, nr_sols, "y-", label="Number of solutions")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.savefig("plot3.png")

        states = np.unique(states, axis=0)
        #print(f"States before filtering: {len(solution_certs)}")
        
        #solutions = [sol for sol in solution_vectors if embedding_check(M, np.array(sol)) == 0]

        #solutions = [sol for sol in solutions if cohom_check(M, np.array(sol), free_size=free_size) == 0]

        #solutions = [sol for sol in solutions if slope_check_real(M, np.array(sol)) == 0]



        #print(f"States after filtering: {len(solutions)}")

        anom_pass = [line for line in states if anomaly_check(M, line) == 0]
        anom_perc = len(anom_pass)/len(states)

        slope_pass = [line for line in states if slope_check_article(M, line) == 0]
        slope_perc = len(slope_pass)/len(states)

        cohom_pass = [line for line in states if cohom_euler_check(M, line, free_size=free_size) == 0]
        cohom_perc = len(cohom_pass)/len(states)

        equiv_pass = [line for line in states if equiv_check(M, line, free_size=free_size) == 0]
        equiv_perc = len(equiv_pass)/len(states)

        

        #df = pd.DataFrame({"Anomaly": anom_pass, "Slope": slope_pass, "Index": cohom_pass, "Equivariance": equiv_pass})


        #columns = df.columns
        #mi_mat = pd.DataFrame(index=columns, columns=columns)

        #for col1 in columns:
        #    h_1 = sk.metrics.mutual_info_score(df[col1], df[col1])
        #    for col2 in columns:
        #        h_2 = sk.metrics.mutual_info_score(df[col2], df[col2])
        #        mi_mat.loc[col1, col2] = sk.metrics.mutual_info_score(df[col1], df[col2])/np.sqrt(h_1*h_2)
        
        #mi_mat = mi_mat.astype(float)
        #plt.figure(figsize=(8, 6))
        #sns.heatmap(mi_mat, annot=True, fmt=".2f", cmap="viridis", square=True)
        #plt.title("Mutual Information Matrix", fontsize=16)
        

        #corr_mat = df.corr()
        #plt.figure(4)
        #sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
        #plt.title("Correlation Matrix", fontsize=16)
        #plt.savefig("correlation_matrix.png")

        bar_names = ["Anomaly", "Slope", "Index", "Equivariance"]

        plt.figure(2)
        plt.bar(bar_names, [anom_perc, slope_perc, cohom_perc, equiv_perc])
        plt.savefig("barplot.png")

        #plt.figure(3)
        #plt.boxplot(t_vec)
        #plt.savefig("timeplot.png")

    



    return solution_vectors

def run_genAlg(x):

    return main()


if __name__ == "__main__":
    n_gen_ep = 30

    episodes = True

    if episodes:
    
        with mp.Pool(processes=10) as pool:
            result_vecs = []
            for result in tqdm(pool.imap_unordered(run_genAlg, range(n_gen_ep)), total=n_gen_ep):
                result_vecs.append(result)

        #result_certs = main()

        total_sols = []
        total_certs = set()
        total_lengths = []
        for i, sol_set in enumerate(result_vecs):
            for s in sol_set:
                #if cohom_check_total(M, s, free_size=free_size) == 0:
                graph = solution_to_graph829(s, low=-up*4, up=-low*4)
                cert = pynauty.certificate(graph)
                if cert not in total_certs:
                    total_sols.append(s)
                total_certs.add(cert)
            total_lengths.append(len(total_certs))
            #print(len(total_sols))


        with open("solutions.npy", "wb") as f:
            np.save(f, np.array(total_sols))


        plt.figure(10)
        plt.plot(range(1,n_gen_ep+1), total_lengths)
        plt.savefig("gen_ep3.png")
    
    else:
        main()