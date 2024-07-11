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