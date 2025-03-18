import argparse
import os
import pickle
import shutil
import time
from pprint import pprint
from typing import List, Optional

import ase
import matplotlib.pyplot as plt
import numpy
import yaml
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch
from ase.phonons import Phonons
from mace.calculators import MACECalculator

from mypytools.mpi.phonons import PhononsMPI
from mypytools.mpi.utils import distribute_work, load_mpi, load_print_func, set_num_cpus
from mypytools.pkg_utils.ase_relax_atoms import init_parser, main

if __name__ == "__main__":
    print("WARNING: Due to package reorganization, please don't use this script path, please use mypytools/pkg_utils/ase_relax_atoms.py instead.")
    parser = init_parser()
    args = parser.parse_args()
    main(**vars(args))
