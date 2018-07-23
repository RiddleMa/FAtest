#encoding=utf8
import numpy as np
from FAtest import ObjFunction


class FAIndividual:

    '''
    individual of firefly algorithm
    '''

    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables 维度
        bound: boundaries of variables 范围
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.trials = 0

    def generate(self):
        '''
        generate a random chromsome for firefly algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = ObjFunction.ObjFunction.GrieFunc(
            self.vardim, self.chrom, self.bound)
if __name__ == '__main__':
    bound = np.tile([[-600], [600]], 25)
    print(bound)
    ind = FAIndividual(25,bound)
    ind.generate()
    ind.calculateFitness()
    print(ind.chrom)
    print(ind.fitness)