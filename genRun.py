from deap import base, creator, tools
import random
from pyCICY import CICY
import time
import numpy as np
import matplotlib.pyplot as plt
from fitness_functions import *
from functions import *
import secrets
import multiprocessing as mp
from tqdm import tqdm



M = M_829


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("vector_bundle", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()


low = -2

up = 2

free_size = 3

#How many k's
LINE_SIZE =  M.len

#How many bundles, (5-1)
IND_SIZE = 4

toolbox.register("map", map)
toolbox.register("attribute", random.randint, low, up)


toolbox.register("vec_ind", tools.initRepeat, list, toolbox.attribute, n=LINE_SIZE)
toolbox.register("individual", tools.initRepeat, creator.vector_bundle, toolbox.vec_ind, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", mate_nested)

toolbox.register("mutate", mutate_nested, low=low, up=up)

toolbox.register("select", tools.selTournament, tournsize=2)

cert_func, ev_func = choose_eval_cert(M)

toolbox.register("evaluate", ev_func, M=M, free_size=free_size)

toolbox.register("cert", cert_func, low=-up*4, up=-low*4)


stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("best", np.max)

logbook = tools.Logbook()
logbook.header = "gen", "avg", "best"


def main(num_pop=300, num_gens=300, plot=True):

    seed = secrets.randbits(32)
    random.seed(seed)
    solution_certs = set()
    solution_vectors = []
    nr_sols = []

    pop = toolbox.population(n=num_pop)
    CXPB, MUTPB, NGEN = 0.6, 0.3, num_gens
    elitism_size = 0
    indpb = 0.4
    surv_size = 500

    
    fitnesses = toolbox.map(toolbox.evaluate, pop)



    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    
    for g in range(NGEN):

        survived = tools.selBest(pop, surv_size)
        survived = list(toolbox.map(toolbox.clone, survived))

        offspring = toolbox.select(pop, len(pop) - surv_size)
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

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if current_best == None or fit[0] > current_best:
                current_best = fit[0]
                
            vec_sol = tot_bundle(ind)
            if all(f == 0 for f in fit):
                count_sols += 1
                canonical_cert = toolbox.cert(vec_sol)
                if canonical_cert not in solution_certs:
                    solution_certs.add(canonical_cert)
                    solution_vectors.append(vec_sol)
        
        record = stats.compile(pop)

        current_best = record["best"]
        print(current_best)
        indpb = adaptive_mut(current_best, indpb)

        offspring.extend(map(toolbox.clone, survived))

        pop[:] = offspring
        print(f"Generation {g+1}/{NGEN}")
        
        logbook.record(gen=g, **record)
        nr_sols.append(len(solution_certs))

    if plot:
        gen = logbook.select("gen")
        fit_avg = logbook.select("avg")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_avg, "b-", label="Average Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, nr_sols, "y-", label="Number of solutions")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.savefig("plot.png")

    return solution_vectors

def run_genAlg(x):

    return main()


if __name__ == "__main__":
    n_gen_ep = 30

    episodes = False

    if episodes:
    
        with mp.Pool(processes=10) as pool:
            result_vecs = []
            for result in tqdm(pool.imap_unordered(run_genAlg, range(n_gen_ep)), total=n_gen_ep):
                result_vecs.append(result)

        total_sols = []
        total_certs = set()
        total_lengths = []
        for i, sol_set in enumerate(result_vecs):
            for s in sol_set:
                cert = toolbox.cert(s)
                if cert not in total_certs:
                    total_sols.append(s)
                total_certs.add(cert)
            total_lengths.append(len(total_certs))

        plt.figure(10)
        plt.plot(range(1,n_gen_ep+1), total_lengths)
        plt.savefig("gen_ep.png")
    
    else:
        main()