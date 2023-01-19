from __future__ import absolute_import

import unittest

from .test_api import *
from .test_ete_evol import *
from .test_ete_build import *
from .test_PagelsLambda import *

def run():
    unittest.main()

if __name__ == '__main__':
    run()
