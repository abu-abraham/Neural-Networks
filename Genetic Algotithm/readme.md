## Genetic Algorithm for determining hyper-parameters


Genetic algorithms (GAs) are a heuristic search and optimization technique inspired by natural evolution . One of the major advantages of genetic algorithm based optimization is that the chances of falling in local minima are greatly minimized.

The entire process follows the natural selection process of reproduction. Similar to natural selection, we mutate, reproduce and create new offspring’s of the best/fittest chromosomes. Each permutation of hyper-parameters is known as chromosomes and each attribute, a gene. The number of hidden-units, epochs, learning-rate, no of batches to be split into, and the optimization algorithm was the attributes/genes in a chromosome. Initially, the population is created by randomly picking values for each gene in a chromosome. Each chromosome is then evaluated and the fitness function is recorded. Then, the selection process where the best chromosomes are selected is done. These selected chromosomes are used to create more offspring’s and mutate. This whole process of selection-mutation-breeding is repeated. 
