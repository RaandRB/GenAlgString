from deap import base, creator, tools
import random
from pyCICY import CICY
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from fitness_functions import *
from functions import *
import secrets
import multiprocessing as mp
from tqdm import tqdm



M = M_5302


creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("vector_bundle", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()


low = -4

up = 4

free_size = 4

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
#toolbox.register("mate", tools.cxMessyOnePoint)

toolbox.register("mutate", mutate_nested, low=low, up=up)
#toolbox.register("mutate", mutate_flat,  low=low, up=up)

#toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("select", tools.selTournamentDCD)
#toolbox.register("select", tools.selNSGA2)

cert_func, ev_func = choose_eval_cert(M)

toolbox.register("evaluate", ev_func, M=M, free_size=free_size)

toolbox.register("cert", cert_func, low=-up*4, up=-low*4)

def main(num_pop=300, num_gens=300, plot=True, print_stats=True):

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("best", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "best"

    seed = secrets.randbits(32)
    random.seed(seed)
    solution_certs = set()
    solution_vectors = []
    nr_sols = []
    best_vec = []

    pop = toolbox.population(n=num_pop)
    CXPB, MUTPB, NGEN = 0.6, 0.3, num_gens
    indpb = 0.1

    
    fitnesses = toolbox.map(toolbox.evaluate, pop)



    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    pop = tools.selNSGA2(pop, len(pop))

    for g in range(NGEN):
        assignCrowdingDist(pop)
        offspring = toolbox.select(pop, len(pop))
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
            if fit[1] == 5:
                count_sols += 1
                canonical_cert = toolbox.cert(vec_sol)
                if canonical_cert not in solution_certs:
                    solution_certs.add(canonical_cert)
                    solution_vectors.append(vec_sol)
        
        record = stats.compile(pop)
        best_vec.append(current_best)

        if print_stats:
           print(f"Generation {g+1}/{NGEN}")
           print(current_best)
        #indpb = adaptive_mut(current_best, indpb)

        #offspring.extend(map(toolbox.clone, survived))

        #pop[:] = offspring


        
        pop[:] = tools.selNSGA2(pop + offspring, len(pop))

        
        logbook.record(gen=g, **record)
        nr_sols.append(len(solution_certs))


    gen = logbook.select("gen")
    fit_avg = logbook.select("avg")

    if plot:


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

    return solution_vectors, best_vec, nr_sols

def run_genAlg(x):

    return main(plot=False, print_stats=False)


if __name__ == "__main__":
    n_gen_ep = 30

    episodes = True

    if episodes:
    
        with mp.Pool(processes=10) as pool:
            result_vecs = []
            for result in tqdm(pool.imap_unordered(run_genAlg, range(n_gen_ep)), total=n_gen_ep):
                result_vecs.append(result)

        fit_vec = [r[1] for r in result_vecs]
        fit_max = np.max(fit_vec, axis=0)
        fit_mean = np.mean(fit_vec, axis=0)
        fit_min = np.min(fit_vec, axis=0)

        nrsols_vec = [r[2] for r in result_vecs]
        nrsols_max = np.max(nrsols_vec, axis=0)
        nrsols_mean = np.mean(nrsols_vec, axis=0)
        nrsols_min = np.min(nrsols_vec, axis=0)

        gen_vec = range(1, len(fit_vec[0]) + 1)

        solutions_vec = [r[0] for r in result_vecs]

        total_sols = []
        total_certs = set()
        total_lengths = []
        for i, sol_set in enumerate(solutions_vec):
            for s in sol_set:
                cert = toolbox.cert(s)
                if cert not in total_certs:
                    total_sols.append(s)
                total_certs.add(cert)
            total_lengths.append(len(total_certs))

        plt.figure(1)
        plt.plot(range(1, len(total_lengths) + 1), total_lengths)
        #plt.savefig("gen_ep.png")

        plt.figure(2)
        plt.fill_between(gen_vec, fit_max, fit_min, alpha=0.3, linewidth=0.5, color='r')
        plt.plot(gen_vec, fit_mean, linewidth=2, color='r')
    
        plt.figure(3)
        plt.fill_between(gen_vec, nrsols_max, nrsols_min, alpha=0.3, linewidth=0.5, color='b')
        plt.plot(gen_vec, nrsols_mean, linewidth=2, color='b')

        converged = [True if nr[-1] > 0 else False for nr in nrsols_vec]
        times = []

        plt.figure(4)
        thresholds = [-125, -100, -75, -50, -25, -15]
        count_conv_upper = np.zeros(len(thresholds))
        count_tot_upper = np.zeros(len(thresholds))
        count_conv_lower = np.zeros(len(thresholds))
        count_tot_lower = np.zeros(len(thresholds))

        for n in range(n_gen_ep):

            fit = fit_vec[n][75]
            
            times.append(fit)

            for i, th in enumerate(thresholds):
                if fit > th:
                    count_tot_upper[i] += 1
                    if converged[n]:
                        count_conv_upper[i] += 1
                elif fit <= th:
                    count_tot_lower[i] += 1
                    if converged[n]:
                        count_conv_lower[i] += 1
                    
        conv_upper = [count_conv_upper[i]/count_tot_upper[i] for i in range(len(thresholds))]
        conv_lower = [count_conv_lower[i]/count_tot_lower[i] for i in range(len(thresholds))]
        count_nonconv = sum([1 for c in converged if not c])
        for i, th in enumerate(thresholds):
            print(f"Converged under {th} at gen 75: {conv_lower[i]*100:0.1f}%")
            print(f"Percentage of non-converged runs contained by threshold: {(count_tot_lower[i] - count_conv_lower[i])/count_nonconv*100:0.1f}%")

        plt.scatter(range(n_gen_ep), times, c=["green" if z else "red" for z in converged], s=80)

        times = []

        plt.figure(5)
        count_conv_upper = np.zeros(len(thresholds))
        count_tot_upper = np.zeros(len(thresholds))
        count_conv_lower = np.zeros(len(thresholds))
        count_tot_lower = np.zeros(len(thresholds))


        for n in range(n_gen_ep):

            fit = fit_vec[n][100]
            
            times.append(fit)

            for i, th in enumerate(thresholds):
                if fit > th:
                    count_tot_upper[i] += 1
                    if converged[n]:
                        count_conv_upper[i] += 1
                elif fit <= th:
                    count_tot_lower[i] += 1
                    if converged[n]:
                        count_conv_lower[i] += 1
                    
        conv_upper = [count_conv_upper[i]/count_tot_upper[i] for i in range(len(thresholds))]
        conv_lower = [count_conv_lower[i]/count_tot_lower[i] for i in range(len(thresholds))]
        count_nonconv = sum([1 for c in converged if not c])
        for i, th in enumerate(thresholds):
            print(f"Converged under {th} at gen 100: {conv_lower[i]*100:0.1f}%")
            print(f"Percentage of non-converged runs contained by threshold: {(count_tot_lower[i] - count_conv_lower[i])/count_nonconv*100:0.1f}%")

        plt.scatter(range(n_gen_ep), times, c=["green" if z else "red" for z in converged], s=80)


        times = []

        plt.figure(6)

        count_conv_upper = np.zeros(len(thresholds))
        count_tot_upper = np.zeros(len(thresholds))
        count_conv_lower = np.zeros(len(thresholds))
        count_tot_lower = np.zeros(len(thresholds))

        for n in range(n_gen_ep):

            fit = fit_vec[n][125]
            
            times.append(fit)

            for i, th in enumerate(thresholds):
                if fit > th:
                    count_tot_upper[i] += 1
                    if converged[n]:
                        count_conv_upper[i] += 1
                elif fit <= th:
                    count_tot_lower[i] += 1
                    if converged[n]:
                        count_conv_lower[i] += 1
                    
        conv_upper = [count_conv_upper[i]/count_tot_upper[i] for i in range(len(thresholds))]
        conv_lower = [count_conv_lower[i]/count_tot_lower[i] for i in range(len(thresholds))]
        count_nonconv = sum([1 for c in converged if not c])
        for i, th in enumerate(thresholds):
            print(f"Converged under {th} at gen 125: {conv_lower[i]*100:0.1f}%")
            print(f"Percentage of non-converged runs contained by threshold: {(count_tot_lower[i] - count_conv_lower[i])/count_nonconv*100:0.1f}%")
        
        plt.scatter(range(n_gen_ep), times, c=["green" if z else "red" for z in converged], s=80)

        #with open("solutionstest.npy", "wb") as f:
        #    np.save(f, total_sols)

        plt.show()
    
    else:
        main()