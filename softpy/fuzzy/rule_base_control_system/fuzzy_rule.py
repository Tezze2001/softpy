from abc import ABC, abstractmethod
from functools import partial
from fuzzyset import FuzzySet, LambdaFuzzySet
import numpy as np

class FuzzyRule(ABC):
    '''
    An abstract class to represent a fuzzy rule.
    '''
    @abstractmethod
    def evaluate(self, params : dict):
        pass

class MamdaniRule(FuzzyRule):
    '''
    An implementation of a fuzzy rule for a Mamdani-type control system.
    The premise of the rule is a dictionary of FuzzySet instances, interpreted as 
    a conjuction obtained using the specified tnorm operation.

    The tnorm itself is a function that constructs a FuzzyCombination object.

    The consequent is a FuzzySet instance.

    The contructor builds the rule as a new FuzzySet (specifically, a FuzzyCombination) 
    by iteratively combining the FuzzySets in the premise
    and then with the consequent.
    '''
    def __init__(self, premise : dict[str, FuzzySet], consequent_name: str, consequent: FuzzySet,
                 tnorm):
        
        if not isinstance(premise, dict):
            raise TypeError("premise should be a dictionary")
        
        if not isinstance(consequent_name, str):
            raise TypeError("consequent_name should be a string")

        for k in premise:
            if not isinstance(premise[k], FuzzySet):
                raise TypeError("All premises should be FuzzySet") 

        if not isinstance(consequent, FuzzySet):
            raise TypeError("consequent should be a FuzzySet")  

        
        self.__premise = premise
        self.__consequent = consequent
        self.__consequent_name = consequent_name
        self.__tnorm = tnorm

        self.__names = np.array(list(self.__premise.keys()))
        curr = self.__premise[self.__names[-1]]
        self.__antecedent = None
        if len(self.__names) > 1:
            for k in self.__names[::-1][1:]:
                self.__antecedent = self.__tnorm(self.__premise[k], curr)
                curr = self.__antecedent
        else:
            self.__antecedent = curr
        self.__rule = self.__tnorm(self.__antecedent, self.__consequent)      
        

    def evaluate(self, params : dict):
        '''
        It evaluates the MamdaniRule at a certain set of values, by computing the corresponding membership degrees: in particular, the evaluate
        method accepts a specification of values for the FuzzySets in the premise and return a LambdaFuzzySet that evaluates the rule at any given
        value of the consequent.
        '''
        if not isinstance(params, dict):
            raise TypeError("params should be a dict or number")
        
        evals = None
        
        if len(self.__names) > 1:
            temp_vals = [None] *self.__names.shape[0]
            
            for k in params.keys():
                idx = np.where(self.__names == k)[0]
                temp_vals[idx] = params[k]

            evals = self.__antecedent(temp_vals)
        else:
            for k in params.keys():
                if k in self.__names:
                    evals = self.__antecedent(params[k])
        return LambdaFuzzySet(lambda u : partial(self.__rule.op, evals)(self.__consequent(u)))