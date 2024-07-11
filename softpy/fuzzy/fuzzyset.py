from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from typing import Callable, Final

def normal_distribution(x: np.number, mean: np.number = 0, var: np.number = 1):
    return 1/np.sqrt(2 * np.pi * var) * np.exp(-((x-mean)**2)/(2*var))

class FuzzySet(ABC):
    '''Abstract Class for a generic fuzzy set'''
    @abstractmethod
    def __call__(self, arg) -> np.number:
        pass

    @abstractmethod
    def __getitem__(self, alpha):
        pass

    @abstractmethod
    def fuzziness(self) -> np.number:
        pass

    @abstractmethod
    def hartley(self) -> np.number:
        pass

class LambdaFuzzySet(FuzzySet):
    ''' Class for a fuzzy set defined by an explicitly specified membership function'''
    

    def __init__(self, 
                 memberships_function: Callable[[np.number], np.number], 
                 epsilon: np.number = 1e-3,
                 bound: tuple[np.number, np.number] = None) -> None:
        self.__epsilon: np.number = 1e-3
        self.__bound: tuple[np.number, np.number]
        self.__memberships_function: Callable[[np.number], np.number]
        self.__f: np.number = -1
        self.__h: np.number = -1
        
        if not isinstance(bound, tuple):
            raise TypeError("Bound should be a tuple that expresses the support " +
                            "of memberships function")

        if (bound[0] > bound[1]):
            raise ValueError("buond[0] should be less equal than buond[1]")
        
        if not isinstance(memberships_function, Callable):
            raise TypeError("Memberships function should be a lambda that takes" + 
                            "as input a real number and returns a value in [0,1]")

        discr_memb_func = np.linspace(bound[0], 
                                      bound[1] + 1, 
                                      int((bound[1] - bound[0] + 1)/epsilon))
        
        discr_memb_func = np.array([0 if 0 <= memberships_function(x) <= 1 else 1 
                                    for x in  discr_memb_func])

        if sum(discr_memb_func) != 0:
            raise ValueError(f"Membership degrees should be floats in [0,1]") 
        
        
        if epsilon >= bound[1] - bound[0] + 1:
            raise ValueError("Epsilon should be small number, ex: 1e-3")

        self.__memberships_function = memberships_function
        self.__epsilon = epsilon
        self.__bound = bound
        self.fuzziness()
        self.hartley()

    @property
    def memberships_function(self) -> Callable:
        return self.__memberships_function

    @property
    def epsilon(self) -> np.number:
        return self.__epsilon
    
    @property
    def bound(self) -> tuple:
        return self.__bound

    def __call__(self, arg: np.number) -> np.number:
        '''
        Compute membership degree of arg parameter.
        '''
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be a number")
        
        if self.__bound[0] <= arg <= self.__bound[1]:
            return self.__memberships_function(arg)
        
        return 0
    
    def __getitem__(self, alpha: np.number) -> np.ndarray:
        '''
        Compute alpha cut of membership function returning an approximation of bounds.
        '''

        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be a float in [0,1], is {type(alpha)}")
        
        step = int((self.__bound[1] - self.__bound[0] + 1)/self.__epsilon)

        x_values = np.linspace(self.__bound[0], 
                               self.__bound[1], 
                               step)
        
        discr_memb_func = np.array([self.__memberships_function(x) for x in  x_values])
        '''
        alpha_cut = (-1, -1)

        for i in range(len(x_values)):
            if alpha_cut[0] == -1 and discr_memb_func[i] >= alpha:
                alpha_cut[0] = x_values[i]
            break
        
        if alpha_cut[0] == -1:
            return ()
        
        for i in range(len(x_values)):
            if alpha_cut[1] == -1 and discr_memb_func[-i] >= alpha:
                alpha_cut[1] = x_values[-i]
            break
        '''
        alpha_cut = np.array([v if self(v) >= alpha else np.nan for v in discr_memb_func])
        return alpha_cut

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LambdaFuzzySet):
            return False

        return (self.__epsilon == other.__epsilon and 
                self.__bound == other.__bound and 
                self.__memberships_function == other.__memberships_function and
                self.__f == other.__f and
                self.__h == other.__h)

    def fuzziness(self) -> np.number:
        return 0
    
    def hartley(self) -> np.number:
        return 0
    '''

    def __fuzzyness_function(self, x):
        if np.abs(x - 0) <= self.__epsilon or np.abs(x - 1) <= self.__epsilon:
            return 0
        return self(x) * np.log2(1/self(x)) + (1 - self(x))*np.log2(1/(1-self(x)))
    def fuzziness(self) -> np.number:
        
        As in the case of alpha cuts, it is generally impossible to compute the 
        fuzziness of a continuous fuzzy set
        analytically: thus, we perform a numerical integration of the fuzziness 
        function between the minimum and maximum
        values of the fuzzy set (it internally uses the __call__ method: notice 
        that it is not implemented in ContinuousFuzzySet!)
        if self.__f == -1:
            self.__f = sp.integrate.quad(self.__fuzzyness_function, 
                                         self.__bound[0], 
                                         self.__bound[1], 
                                         epsabs=self.__epsilon)[0]
        return self.__f

    def hartley(self) -> np.number:
        if self.__h == -1:
            self.__h = sp.integrate.quad(lambda x: np.log2(self[x][0][1] - self[x][0][0]),
                                         0, 
                                         1)[0]   
        return self.__h
        '''

class DiscreteFuzzySet(FuzzySet):
    '''
    Implements a discrete fuzzy set
    
    '''

    def __init__(self, items, memberships, dynamic=True):
        '''
        Requires as input a sequence (list or array) or objects and a sequence (list or array) of membership degrees
        Attribute dynamic controls whether the support set is exhaustive or not (i.e. there exist objects not in items
        whose membership degree is 0)

        Internally the constructor uses a dictionary (self.set) to enable fast look-up of membership degrees
        '''
        if type(items) != list and type(items) != np.ndarray:
            raise TypeError("items should be list or numpy.array")
        
        if type(memberships) != list and type(memberships) != np.ndarray:
            raise TypeError("memberships should be list or numpy.array")
        
        if type(dynamic) != bool:
            raise TypeError("dynamic should be bool")
        
        self.items = np.array(items)
        self.set = dict(zip(items, range(len(items))))

        for m in memberships:
            if not np.issubdtype(type(m), np.number):
                raise TypeError("Membership degrees should be floats in [0,1], is %s" % type(m))
            if  m < 0 or m > 1:
                raise ValueError("Membership degrees should be floats in [0,1], is %s" % m)
            
        self.memberships = np.array(memberships)
        self.dynamic = dynamic

    def __call__(self, arg):
        '''
        Gets the membership degree of arg. Uses self.set to enable quick look-up.
        Behavior changes according to value of dynamic
        '''
        if arg not in self.set.keys():
            if self.dynamic:
                self.set[arg] = len(self.items)
                self.items = np.append(self.items, arg)
                self.memberships = np.append(self.memberships, 0.0)
            else:
                raise ValueError("%s not in the support of the fuzzy set" % arg)
             
        return self.memberships[self.set[arg]]
        
    def __getitem__(self, alpha: np.number) -> np.ndarray:
        '''
        Gets an alpha cut
        '''
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        return self.items[self.memberships >= alpha]
    
    def __eq__(self, other: object) -> bool:
        '''
        Checks whether two DiscreteFuzzySet instances are equal
        '''
        if not isinstance(other, DiscreteFuzzySet):
            return NotImplemented
        
        for v in list(self.set.keys()) + list(other.set.keys()):
            try:
                v1 = self(v)
            except ValueError:
                return False
            try:
                v2 = other(v)
            except ValueError:
                return False
            
            if v1 != v2:
                return False
        return  True
    
    def fuzziness(self) -> np.number:
        '''
        Computes the fuzziness
        '''
        try:
            return self.f
        except AttributeError:
            pos = self.memberships[self.memberships > 0]
            pos = pos*np.log2(1/pos)
            non = self.memberships[self.memberships < 1]
            non = (1-non)*np.log2(1/(1-non))
            self.f : np.number = np.sum(pos) + np.sum(non)
            return self.f
        
    def hartley(self) -> np.number:
        '''
        Computes the hartley entropy (non-specificity)
        '''
        try:
            return self.h
        except AttributeError:
            pos = self.memberships[self.memberships > 0]
            sort = np.append(np.sort(pos)[::-1], 0)
            coeffs = sort[:-1] - sort[1:]
            sizes = np.log2(np.array([len(self[i]) for i in sort[:-1]]))
            self.h : np.number = np.sum(coeffs*sizes)
            return self.h
    

class ContinuousFuzzySet(FuzzySet):
    '''
    Abstract class for a continuous fuzzy set
    Note: each (bounded) continuous fuzzy set has a minimum and maximum elements s.t. their membership degrees are > 0
    Epsilon is used to define an approximation degree for various operations that cannot be implemented exactly 
    '''
    min : np.number
    max : np.number
    epsilon = 0.01

    def __getitem__(self, alpha: np.number) -> np.ndarray | tuple:
        '''
        In general, it is not possible to take alpha cuts of continuous fuzzy sets analytically: we need to search
        explicitly for all values whose membership degree is >= alpha. Since it is impossible to look all real numbers,
        we do a grid search where the grid step-size is defined by epsilon
        '''

        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        
        
        grid = np.linspace(self.min, self.max, int((self.max - self.min)/self.epsilon))
        vals = np.array([v if self(v) >= alpha else np.nan for v in grid])
        return vals
    
    def fuzziness(self) -> np.number:
        '''
        As in the case of alpha cuts, it is generally impossible to compute the fuzziness of a continuous fuzzy set
        analytically: thus, we perform a numerical integration of the fuzziness function between the minimum and maximum
        values of the fuzzy set (it internally uses the __call__ method: notice that it is not implemented in ContinuousFuzzySet!)
        '''
        try:
            return self.f
        except AttributeError:
            func_int = lambda x: 0 if (np.abs(x - 0) <= self.epsilon or np.abs(x - 1) <= self.epsilon) else self(x)*np.log(1/self(x)) + (1 - self(x))*np.log2(1/(1-self(x)))
            self.f : np.number = sp.integrate.quad(func_int, self.min, self.max, epsabs=self.epsilon)[0]
            return self.f


class FuzzyNumber(ContinuousFuzzySet):
    '''
    Abstract class for a fuzzy number (convex, closed fuzzy set over the real numbers)
    '''

    def hartley(self) -> np.number:
        '''
        In the case of fuzzy numbers it is easy to compute the hartley entropy, because we know that each
        alpha cut is an interval. Thus, for each alpha, we simply compute the (logarithm of the) length
        of the corresponding interval and then integrate over alpha
        '''
        try:
            return self.h
        except AttributeError:
            func_int = lambda x: np.log2(self[x][1] - self[x][0])
            self.h : np.number = sp.integrate.quad(func_int, 0, 1)[0]
            return self.h 


class IntervalFuzzyNumber(FuzzyNumber):
    '''
    Implements an interval fuzzy number (equivalently, an interval)
    '''
    def __init__(self, lower: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError("Lower should be float, is %s" % type(lower))
        
        if not np.issubdtype(type(upper), np.number):
            raise TypeError("Upper should be float, is %s" % type(upper))
        
        if lower > upper:
            raise ValueError("Lower should be smaller than Upper")
        
        self.lower = lower
        self.upper = upper
        self.min = lower
        self.max = upper

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        if arg < self.lower or arg > self.upper:
            return 0.0
        else:
            return 1.0
        
    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        return self.lower, self.upper
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalFuzzyNumber):
            return NotImplemented
        return self.lower == other.lower and self.upper == other.upper
    
    def fuzziness(self) -> np.number:
        return 0.0
    
    def hartley(self) -> np.number:
        return np.log2(self.upper - self.lower)
    


class RampFuzzyNumber(FuzzyNumber):
    '''
    Implements a ramp fuzzy number
    '''
    def __init__(self, lower: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError("Lower should be float, is %s" % type(lower))
        
        if not np.issubdtype(type(upper), np.number):
            raise TypeError("Upper should be float, is %s" % type(upper))
        
        self.lower = lower
        self.upper = upper
        self.min = np.min([lower,upper])
        self.max = np.max([lower,upper])

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        if self.lower <= self.upper:
            if arg <= self.lower:
                return 0
            elif arg >= self.upper:
                return 1
        else:
            if arg >= self.lower:
                return 0
            elif arg <= self.upper:
                return 1
               
        return (arg - self.lower)/(self.upper - self.lower)
        
        
    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        x = (self.upper - self.lower)*alpha + self.lower
        if self.upper > self.lower:
            return x, self.upper
        else:
            return self.upper, x
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RampFuzzyNumber):
            return NotImplemented
        return self.lower == other.lower and self.upper == other.upper
    
    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            func = lambda x: (x - self.lower)/(self.upper - self.lower)
            func_int = lambda x: func(x)*np.log(1/func(x)) + (1 - func(x))*np.log2(1/(1-func(x)))
            self.f : np.number = sp.integrate.quad(func_int, self.lower, self.upper)[0]
            return self.f




class TriangularFuzzyNumber(FuzzyNumber):
    '''
    Implements a triangular fuzzy number
    '''
    def __init__(self, lower: np.number, middle: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError("Lower should be floats, is %s" % type(lower))
        
        if not np.issubdtype(type(middle), np.number):
            raise TypeError("Middle should be floats, is %s" % type(middle))

        if not np.issubdtype(type(upper), np.number):
            raise TypeError("Higher should be floats, is %s" % type(upper))
        
        if lower > middle or lower > upper:
            raise ValueError("Lower should be smaller than Middle and Upper")
        
        if middle > upper:
            raise ValueError("Middle should be smaller than Upper")
        
        self.lower = lower
        self.middle = middle
        self.upper = upper

        self.min = lower
        self.max = upper

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        if arg < self.lower or arg > self.upper:
            return 0
        elif arg >= self.lower and arg <= self.middle:
            return (arg - self.lower)/(self.middle - self.lower)
        else:
            return (self.upper - arg)/(self.upper - self.middle)
        
    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        low = alpha*(self.middle - self.lower) + self.lower
        upp = self.upper - alpha*(self.upper - self.middle)
        
        return low, upp
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TriangularFuzzyNumber):
            return NotImplemented
        return self.lower == other.lower and self.middle == other.middle and self.upper == other.upper
    
    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            left = lambda x: (x - self.lower)/(self.middle - self.lower)
            right = lambda x: (self.upper - x)/(self.upper - self.middle)
            left_int = lambda x: left(x)*np.log(1/left(x)) + (1 - left(x))*np.log2(1/(1-left(x)))
            right_int = lambda x: right(x)*np.log(1/right(x)) + (1 - right(x))*np.log2(1/(1-right(x)))
            self.f : np.number = sp.integrate.quad(left_int, self.lower, self.middle)[0]
            self.f += sp.integrate.quad(right_int, self.middle, self.upper)[0]
            return self.f
        
class TrapezoidalFuzzyNumber(FuzzyNumber):
    '''
    Implements a trapezoidal fuzzy number 
    '''
    def __init__(self, lower: np.number, middle1: np.number, middle2: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError("Lower should be floats, is %s" % type(lower))
        
        if not np.issubdtype(type(middle1), np.number):
            raise TypeError("Middle should be floats, is %s" % type(middle1))
        
        if not np.issubdtype(type(middle2), np.number):
            raise TypeError("Middle should be floats, is %s" % type(middle2))

        if not np.issubdtype(type(upper), np.number):
            raise TypeError("Higher should be floats, is %s" % type(upper))
        
        if lower > middle1 or lower > middle2 or lower > upper:
            raise ValueError("Lower should be the smallest input")
        
        if middle1 > middle2 or middle1 > upper:
            raise ValueError("Middle1 should be smaller than Middle2 and Upper")
        
        if middle2 > upper:
            raise ValueError("Upper should be the largest input")
        
        self.lower = lower
        self.middle1 = middle1
        self.middle2 = middle2
        self.upper = upper

        self.min = lower
        self.max = upper

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        if arg < self.lower or arg > self.upper:
            return 0
        elif self.lower <= arg  <= self.middle1:
            return (arg - self.lower)/(self.middle1 - self.lower)
        elif self.middle1 <= arg <= self.middle2:
            return 1
        else:
            return (self.upper - arg)/(self.upper - self.middle2)
        
    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        low = alpha*(self.middle1 - self.lower) + self.lower
        upp = alpha*(self.upper - self.middle2) + self.middle2
        
        return low, upp
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrapezoidalFuzzyNumber):
            return NotImplemented
        return self.lower == other.lower and self.middle1 == other.middle1 and self.middle2 == other.middle2 and self.upper == other.upper
    
    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            left = lambda x: (x - self.lower)/(self.middle1 - self.lower)
            right = lambda x: (self.upper - x)/(self.upper - self.middle2)
            left_int = lambda x: left(x)*np.log(1/left(x)) + (1 - left(x))*np.log2(1/(1-left(x)))
            right_int = lambda x: right(x)*np.log(1/right(x)) + (1 - right(x))*np.log2(1/(1-right(x)))
            self.f : np.number = sp.integrate.quad(left_int, self.lower, self.middle1)[0]
            self.f += sp.integrate.quad(right_int, self.middle2, self.upper)[0]
            return self.f
        



class GaussianFuzzyNumber(FuzzyNumber):
    '''
    Implements a Gaussian fuzzy number
    '''
    def __init__(self, mean: np.number, std: np.number):
        if not np.issubdtype(type(mean), np.number):
            raise TypeError("Mean should be float, is %s" % type(mean))
        
        if not np.issubdtype(type(std), np.number):
            raise TypeError("Middle should be floats, is %s" % type(std))
        
        if std < 0:
            raise ValueError("std should be positive")
        
        self.mean = mean
        self.std = std

        self.min = self.mean - np.sqrt(2*self.std**2 * np.log(1/0.001))
        self.max = self.mean + np.sqrt(2*self.std**2 * np.log(1/0.001))

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        return np.exp(-(arg - self.mean)**2/(2*self.std**2))
        
    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        low = self.mean - np.sqrt(2*self.std**2 * np.log(1/alpha))
        upp = self.mean + np.sqrt(2*self.std**2 * np.log(1/alpha))
        
        return low, upp
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussianFuzzyNumber):
            return NotImplemented
        return self.mean == other.mean and self.std == other.std
    
    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            intgr = lambda x: self(x)*np.log(1/self(x)) + (1 - self(x))*np.log2(1/(1-self(x)))
            self.f : np.number = sp.integrate.quad(intgr, self.min, self.max)[0]
            return self.f