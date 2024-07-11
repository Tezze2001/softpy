import sys
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/..")

from softpy.fuzzy.fuzzyset import LambdaFuzzySet, normal_distribution

@contextmanager
def not_raises():
    try:
        yield
        
    except Exception as err:
        raise AssertionError(
            # "Did raise exception {0} when it should not!".format(
                
            # )
            repr(err)
        )

class TestLambdaFuzzySet:
    @pytest.mark.parametrize(
            "labda_member,epsilon,bound,exception_expected", 
            [
                (lambda x: 1 if x == 0 else 0, 1e-3, (0, 0), None), #check membership function
                (1, 1e-3, (0, 0), TypeError),
                (lambda x: 2 if x == 0 else 0, 1e-3, (0, 0), ValueError), 
                (lambda x: 1 if x == 0 else -1, 1e-3, (0, 0), ValueError),
                (normal_distribution, 1e-3, (0, 1), None),
                (normal_distribution, (0,1), (0, 1), TypeError), #check epsilon
                (normal_distribution, 2, (0, 1), ValueError),
                (normal_distribution, -1e-3, (0, 1), ValueError),
                (normal_distribution, 1e-3, 2, TypeError), #check bound
                (normal_distribution, 1e-3, (1, 0), ValueError)
            ])
    def test_creation(self,
                      labda_member: Callable, 
                      epsilon: np.number, 
                      bound: tuple, 
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = LambdaFuzzySet(labda_member, epsilon, bound)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = LambdaFuzzySet(labda_member, epsilon, bound)

    @pytest.mark.parametrize(
            "fuzzy_set,alpha_cut,exception_expected", 
            [
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.1, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.2, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.3, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.4, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.5, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.6, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.7, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.7, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.8, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.9, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 0.1, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 2, ValueError),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 'a', TypeError),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), 2.2, ValueError),
                (LambdaFuzzySet(normal_distribution, 1e-3, (0, 1)), -1.1, ValueError),
            ])
    def test_alpha_cut(self, 
                       fuzzy_set: LambdaFuzzySet, 
                       alpha_cut: tuple,
                       exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                step = int((fuzzy_set.bound[1] - fuzzy_set.bound[0] + 1)/fuzzy_set.epsilon)

                x_values = np.linspace(fuzzy_set.bound[0], 
                                       fuzzy_set.bound[1], 
                                       step)
                
                discr_memb_func = np.array([fuzzy_set.memberships_function(x) for x in  x_values])
                alpha_cut_set = np.array([v if fuzzy_set(v) >= alpha_cut else np.nan for v in discr_memb_func])
                assert np.array_equal(alpha_cut_set, fuzzy_set[alpha_cut], equal_nan=True)
        else:
            with pytest.raises(exception_expected) as e_info:
                alpha_cut_set = fuzzy_set[alpha_cut]


    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 'a', TypeError),
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 3, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 0, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 1, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 1.2, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 1.5, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 1.7, None),
                (LambdaFuzzySet(normal_distribution, 1e-3, (1, 2)), 2, None),
            ]
            )
    def test_memberships(self, 
                         fuzzy_set: LambdaFuzzySet, 
                         arg: tuple,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                print(fuzzy_set.bound[0] <= arg <= fuzzy_set.bound[1])
                if fuzzy_set.bound[0] <= arg <= fuzzy_set.bound[1]:
                    assert (fuzzy_set(arg) - (fuzzy_set.memberships_function)(arg)) <= fuzzy_set.epsilon
                else:
                    assert fuzzy_set(arg) - 0.0 <= fuzzy_set.epsilon
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)