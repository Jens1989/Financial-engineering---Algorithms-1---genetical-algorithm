# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:56:02 2018

@author: jens_
"""

# --- IMPORT LIBRARIES --------------------------------------------------------

import datetime                     # Import datetime module 
import matplotlib.pyplot as plt     # Import matplotlib ploting library  
import numpy as np                  # Import numpy
import random                       # Import random 
import pandas as pd

# --- GOBAL PARAMETERS --------------------------------------------------------

ALLELE_SIZE = 23        # Number of allele + 1 
POPULATION_SIZE = 8    # Population size
ELITE_CHROMOSOMES = 0  # Elite chromosome not subjected to evolving(cross-over, mutation) from one generation to the next 
SELECTION_SIZE = 10     # Selection size randomly from population
MUTATION_RATE = 0.25   # Probability that chromosome gene does a random mutation
TARGET_FITNESS = 3   # Target fitness (change as need to get convergence)
ALPHA = 0.1            # Constant in the fitness test


# Random number generator list (change as needed)
R = [x+15 for x in range(0,360,15)]

# Data collection

data = pd.read_csv(r'C:\Users\jens_\Dropbox\Jens\financial engineering\algorithms 1\miniproject 3\TSLA.csv')

# Assest closing price and standard deviation
PRICE  = data["Adj Close"].values.tolist()
SIGMA0 = np.std(PRICE) # Standard deviation of the entire timeseries

# --- CLASSES -----------------------------------------------------------------

class Chromosome:
     
    """
    Candidate solution.
    """
     
    def __init__(self):
        self._genes = []
        self._fitness = 0        
        self._genes.append(random.randint(R[0],R[1]))
        self._genes.append(random.randint(R[1],R[2]))
        self._genes.append(random.randint(R[2],R[3]))
        self._genes.append(random.randint(R[3],R[4]))
        self._genes.append(random.randint(R[4],R[5]))
        self._genes.append(random.randint(R[5],R[6]))
        self._genes.append(random.randint(R[6],R[7]))
        self._genes.append(random.randint(R[7],R[8]))
        self._genes.append(random.randint(R[8],R[9]))
        self._genes.append(random.randint(R[9],R[10]))
        self._genes.append(random.randint(R[10],R[11]))
        self._genes.append(random.randint(R[11],R[12]))
        self._genes.append(random.randint(R[12],R[13]))
        self._genes.append(random.randint(R[13],R[14]))
        self._genes.append(random.randint(R[14],R[15]))
        self._genes.append(random.randint(R[15],R[16]))
        self._genes.append(random.randint(R[16],R[17]))
        self._genes.append(random.randint(R[17],R[18]))
        self._genes.append(random.randint(R[18],R[19]))
        self._genes.append(random.randint(R[19],R[20]))
        self._genes.append(random.randint(R[20],R[21]))
        self._genes.append(random.randint(R[21],R[22]))
        self._genes.append(random.randint(R[22],R[23]))

 
    def get_genes(self):
        return self._genes
     
    def get_fitness(self):
        L = []
        D = []
        for i in range(len(PRICE)-2):
            if self._genes[0] <= PRICE[i]:
                if self._genes[1] <= PRICE[i+1] or PRICE[i+1] <= self._genes[2]:
                    if self._genes[3] <= PRICE[i+2] or PRICE[i+2] <= self._genes[4]:
                        L.append(PRICE[i+1])
                        D.append(i)   
        sigma = np.std(L)
        nc = len(L)
        self._fitness = -np.log2(sigma/SIGMA0)-ALPHA/nc        
        return self._fitness
 
    def __str__(self):
        return self._genes.__str__()
    
#    def _str_(self):
#        return " ".join(_genes)

class Population:
     
    """
    Pupulation of candidate solutions.
    """
     
    def __init__(self, size):
        self._chromosomes = []
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome())
            i += 1
     
    def get_chromosomes(self):
        return self._chromosomes
    
    def print_chromosomes(self):
        print ([i._genes for i in self._chromosomes])


class GeneticAlgorithm:
     
    """
    The genectic algorithm logic for evolving the population via crossover and
    mutation.
    """
     
    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm._mutate_population(pop)
     
    @staticmethod
    def _crossover_population(pop):
        crossover_pop = Population(0)
        for i in range(ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm._select_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm._select_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
            i += 1
        return crossover_pop
     
    @staticmethod
    def _mutate_population(pop):
        crossover_pop = GeneticAlgorithm._crossover_population(pop)
        final_pop = Population(0)
        for i in range(ELITE_CHROMOSOMES, POPULATION_SIZE):
            random_mutation = random.randint(0,4)
            if random_mutation == 0: # no mutation
                final_pop.get_chromosomes().append(crossover_pop.get_chromosomes()[i])
            elif random_mutation == 1: # add a new condition
                crossover_pop.get_chromosomes()[i].get_genes().insert(random.randint(0, len(crossover_pop.get_chromosomes()[i].get_genes())),random.randint(R[0],R[-1]))
                final_pop.get_chromosomes().append(crossover_pop.get_chromosomes()[i])
            elif random_mutation == 2 : # delete a condition
                crossover_pop.get_chromosomes()[i].get_genes().pop(random.randint(0, len(crossover_pop.get_chromosomes())))
                final_pop.get_chromosomes().append(crossover_pop.get_chromosomes()[i])
            elif random_mutation == 3 : # shift a range up or down
                crossover_pop.get_chromosomes()[i].get_genes()[-1] = crossover_pop.get_chromosomes()[i].get_genes()[random.randint(0, len(crossover_pop.get_chromosomes()))]+random.randint(-50,50)
                final_pop.get_chromosomes().append(crossover_pop.get_chromosomes()[i])
            else: # broaden or shrinken a range
                change_value = random.randint(-50,50)
                crossover_pop.get_chromosomes()[i].get_genes()[random.randint(0, len(crossover_pop.get_chromosomes()))] = crossover_pop.get_chromosomes()[i].get_genes()[random.randint(0, len(crossover_pop.get_chromosomes()))]+change_value
                crossover_pop.get_chromosomes()[i].get_genes()[random.randint(0, len(crossover_pop.get_chromosomes()))] = crossover_pop.get_chromosomes()[i].get_genes()[random.randint(0, len(crossover_pop.get_chromosomes()))]-change_value
                final_pop.get_chromosomes().append(crossover_pop.get_chromosomes()[i])
        return final_pop
     
    @staticmethod
    #does random gene selection from each of parent chromosomes
    def _crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Chromosome()
        crossover_chrom.get_genes().clear()
        lchrom1 = len(chromosome1.get_genes())
        lchrom2 = len(chromosome2.get_genes())
        iterator = 0
        while lchrom1 > 0 and lchrom2 > 0:
                if random.random() < 0.5:
                    crossover_chrom.get_genes().append(chromosome1.get_genes()[iterator])
                    lchrom1 -=1
                    lchrom2 -=1
                    iterator +=1
                else:
                    crossover_chrom.get_genes().append(chromosome2.get_genes()[iterator])
                    lchrom1 -=1
                    lchrom2 -=1
                    iterator +=1
        return crossover_chrom
         
    @staticmethod
    #Select the one with the highest fitness 
    def _select_population(pop):
        select_pop = Population(0)
        i = 0
        while i < SELECTION_SIZE:
            select_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0,POPULATION_SIZE)])
            i += 1
#        Sort by fitness, highest is the first    
        select_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        return select_pop

def print_population(pop, gen_number):
     
    """
    Print evolving generation results.
    """
     
    print("\n-----------------------------------------------------------")
    print("Generation #", gen_number, ": Fittest chromosome fitness: %3.2f" % pop.get_chromosomes()[0].get_fitness())
    print("-----------------------------------------------------------")
    i = 0
    for x in pop.get_chromosomes():
        print("Chromosome #", i, " :", x, "| Fitness: %3.2f" % x.get_fitness())
        i += 1

# evaluate the population
population = Population(POPULATION_SIZE)
population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True) #Population with the highest fitness is the first
print_population(population, 0)

#While the fitness of the fittest chromosome in each population is smaller than target fitness
# evolve the population and sort the evolved the population, print it and go to another population
  
gen = 0
while population.get_chromosomes()[0].get_fitness() < TARGET_FITNESS:
    population = GeneticAlgorithm.evolve(population)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    print_population(population, gen)
    gen += 1


            