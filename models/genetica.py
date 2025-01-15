"""
Filename: genetica.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""
import pygad
import logging
import numpy
from models.model import Model
from utils.language_helpers import num_to_seq,seq_to_num,mutation_callback,is_valid_codon_seq
class Genetica(Model):
    """
    genetica类，用于对基因利用pygad遗传算法进行突变和筛选
    """
    def __init__(self, config,classifier,initial_population=None,codon_part_sequence=None):
        self.EPSILON = 1e-6
        self.classifier = classifier
        self.num_generations = 50# Number of generations.
        self.num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.
        self.sol_per_pop = 8# Number of solutions in the population.
        self.parent_selection_type = "sss"
        self.keep_parents = 1
        self.crossover_type = "single_point"
        self.mutation_type = "random"
        self.mutation_percent_genes = 10
        self.config=config
        self.initial_population = initial_population
        self.last_fitness = 0
        self.read_config(self.config)
        if self.type == 'rbs' and codon_part_sequence!=None:
            with open(self.rbs_library_path) as f:
                self.initial_population = []
                initial_populations = f.readlines()
                for initial_population in initial_populations:
                    seq = seq_to_num(initial_population.replace('\n', ''))
                    chromosome = []
                    for i in range(len(seq)):
                        chromosome.append(seq[i])
                    self.initial_population.append(chromosome)
            f.close()
            self.codon_part_sequence = codon_part_sequence
            self.ga_instance = pygad.GA(num_generations=self.num_generations,
                    num_parents_mating=self.num_parents_mating,
                    fitness_func=self.rbs_fitness_func,
                    parent_selection_type=self.parent_selection_type,
                    keep_parents=self.keep_parents,
                    crossover_type=self.crossover_type,
                    mutation_type=self.mutation_type,
                    mutation_percent_genes=self.mutation_percent_genes,
                    on_generation=self.on_generation,
                    initial_population=self.initial_population,
                    gene_type=int,
                    gene_space=[1,2,3,4],
                    )
        else:
            self.codon_part_sequence=None
            self.ga_instance = pygad.GA(num_generations=self.num_generations,
                        num_parents_mating=self.num_parents_mating,
                        fitness_func=self.fitness_func,
                        parent_selection_type=self.parent_selection_type,
                        keep_parents=self.keep_parents,
                        crossover_type=self.crossover_type,
                        mutation_type=self.mutation_type,
                        mutation_percent_genes=self.mutation_percent_genes,
                        on_generation=self.on_generation,
                        initial_population=self.initial_population,
                        gene_type=int,
                        gene_space=[1,2,3,4],
                        save_solutions=self.type=="codon"
                        )

    def read_config(self,config):
        self.expected_result =  config["expected_result"]
        self.num_generations = config["num_generations"]
        self.num_parents_mating = config["num_parents_mating"]
        self.parent_selection_type = config["parent_selection_type"]
        self.keep_parents = config["keep_parents"]
        self.crossover_type = config["crossover_type"]
        self.mutation_type = config["mutation_type"]
        self.mutation_percent_genes =config["mutation_percent_genes"]
        self.plot_fitness = config["plot_fitness"]
        self.type = config["type"]
        if self.type == 'rbs':
            self.rbs_library_path = config["rbs_library_path"]

    def main(self):
        # Running the GA to optimize the parameters of the function.
        self.ga_instance.run()
        if self.plot_fitness:
            self.ga_instance.plot_fitness()
        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution(self.ga_instance.last_generation_fitness)
        solution = num_to_seq(solution)
        if self.type == 'rbs' and self.codon_part_sequence==None:
            if not is_valid_codon_seq(solution):
                solutions = self.ga_instance.solutions
                best_fitness = 0
                generation_idx = 0
                for i in range(len(solutions)):
                    solution_seq = num_to_seq(solutions[-i])
                    if is_valid_codon_seq(solution_seq):
                        output,acc = self.classifier.predict([solution_seq])
                        fitness = 1.0 / (numpy.abs(output - self.expected_result) + self.EPSILON)
                        if fitness > best_fitness:
                            solution = solution_seq
                            best_fitness = fitness
                            generation_idx = len(solutions)-i
                    elif i == len(solutions):
                            logging.info(f"No solution is a valid codon sequence!")
                            return 0
                logging.info(f"Parameters of the best solution : {solution}")
                self.save_solution = solution
                logging.info(f"Fitness value of the best solution = {best_fitness}")
                logging.info(f"Generation of the best solution : {generation_idx}")
                prediction,acc = self.classifier.predict([solution])
                logging.info(f"Predicted output based on the best solution : {prediction}")
                return prediction == self.expected_result
        if self.type == 'rbs'and self.codon_part_sequence!=None:
            self.save_solution = solution + self.codon_part_sequence
        else:
            self.save_solution = solution
        logging.info(f"Parameters of the best solution : {self.save_solution}")
        logging.info(f"Fitness value of the best solution = {solution_fitness}")
        logging.info(f"Index of the best solution : {solution_idx}")
        if self.ga_instance.best_solution_generation != -1:
            logging.info(f"Best fitness value reached after {self.ga_instance.best_solution_generation} generations.")
        prediction,acc = self.classifier.predict([self.save_solution])
        logging.info(f"Predicted output based on the best solution : {prediction}")
        return prediction == self.expected_result

    def train(self):
        # Saving the GA instance.
        self.save_genetica_path=self.config["save_genetica_path"]
        self.save_solution_path = self.config["save_solution_path"]
        self.ga_instance.save(filename=self.save_genetica_path)
        with open(self.save_solution_path,'a+') as f:
            f.write(self.save_solution+'\n')
        f.close()
        

    def predict(self):
    # Loading the saved GA instance.
        self.load_genetica_path = self.config["load_genetica_path"]
        self.ga_instance = pygad.load(filename=self.load_genetica_path)
        if self.plot_fitness:
            self.ga_instance.plot_fitness()

    def on_generation(self,ga_instance):
        # logging.info(f"Parameters of the best solution : {num_to_seq(ga_instance.best_solution(ga_instance.last_generation_fitness)[0])}")
        # logging.info(f"Generation = {ga_instance.generations_completed}")
        # logging.info(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
        # logging.info(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness}")
        self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

    def fitness_func(self,ga_instance,solution, solution_idx):
        output,acc = self.classifier.predict([num_to_seq(solution)])
        fitness = 1.0 / (numpy.abs(output - self.expected_result) + self.EPSILON) 
        return fitness
    
    def rbs_fitness_func(self,ga_instance,solution, solution_idx):
        solution = num_to_seq(solution)
        solution += self.codon_part_sequence
        output,acc = self.classifier.predict([solution])
        fitness = 1.0 / (numpy.abs(output - self.expected_result) + self.EPSILON) 
        return fitness