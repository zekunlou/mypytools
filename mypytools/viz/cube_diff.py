"""
get the difference between two cube files and write to a new cube file
"""

import argparse
import os

import numpy
from ase.io.cube import read_cube, write_cube


def get_diff(in1: str, in2: str, out: str):
    # read cube files
    in1 = open(in1)
    in2 = open(in2)
    cube1 = read_cube(in1)
    cube2 = read_cube(in2)
    in1.close()
    in2.close()
    assert numpy.allclose(cube1["origin"], cube2["origin"])
    diff = cube1["data"] - cube2["data"]
    # write cube file
    out = open(out, "w")
    write_cube(out, cube1["atoms"], diff, origin=cube1["origin"])
    out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="get the difference between two cube files and write to a new cube file"
    )
    parser.add_argument("-f1", "--file1", type=str, help="cube file 1, file1 - file2")
    parser.add_argument("-f2", "--file2", type=str, help="cube file 2, file1 - file2")
    parser.add_argument("-o", "--output", type=str, help="output file name")
    args = parser.parse_args()

    fpaths = {
        "in1": os.path.join(os.getcwd(), args.file1),
        "in2": os.path.join(os.getcwd(), args.file2),
        "out": os.path.join(os.getcwd(), args.output),
    }

    for key in fpaths:
        if key == "out":
            continue
        if not os.path.exists(fpaths[key]):
            raise FileNotFoundError(fpaths[key])

    get_diff(**fpaths)
