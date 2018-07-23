import numpy as np
from FAtest import FAIndividual
import random
import copy
import matplotlib.pyplot as plt


class FireflyAlgorithm:
    '''
    The class for firefly algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop 总数
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition 最大循环次数
        param: algorithm required parameters, it is a list which is consisting of [beta0, gamma, alpha]
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = FAIndividual.FAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):
        '''
        evaluation of the population fitnesses
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        '''
        evolution process of firefly algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluate()
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.move()
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:#更新最优点
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        self.printResult()
        #########################
        print(self.best.fitness)

    def move(self):
        '''
        move the a firefly to another brighter firefly
        '''
        for i in range(0, self.sizepop):
            for j in range(0, self.sizepop):
                if self.fitness[j] > self.fitness[i]:
                    r = np.linalg.norm(
                        self.population[i].chrom - self.population[j].chrom)
                    beta = self.params[0] * \
                           np.exp(-1 * self.params[1] * (r ** 2))
                    # beta = 1 / (1 + self.params[1] * r)
                    # print beta
                    self.population[i].chrom += beta * (self.population[j].chrom - self.population[
                        i].chrom) + self.params[2] * np.random.uniform(low=-1, high=1, size=self.vardim)
                    for k in range(0, self.vardim):
                        if self.population[i].chrom[k] < self.bound[0, k]:
                            self.population[i].chrom[k] = self.bound[0, k]
                        if self.population[i].chrom[k] > self.bound[1, k]:
                            self.population[i].chrom[k] = self.bound[1, k]
                    self.population[i].calculateFitness()
                    self.fitness[i] = self.population[i].fitness

    def printResult(self):
        '''
        plot the result of the firefly algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Firefly Algorithm for function optimization")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    bound = np.tile([[-600], [600]], 25)
    fa = FireflyAlgorithm(60, 25, bound, 200, [1.0, 0.000001, 0.6])
    fa.solve()
