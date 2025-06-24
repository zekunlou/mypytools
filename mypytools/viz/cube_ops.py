"""
get the difference between two cube files and write to a new cube file
"""

import argparse
import os

import numpy
from ase.io.cube import read_cube, write_cube


def cube_operation(in1: str, in2: str, out: str, op: str):
    assert op in ("sub", "add", "mul", "div"), f"operation {op} not supported"
    # read cube files
    in1 = open(in1)
    cube_in1 = read_cube(in1)
    in1.close()
    in2 = open(in2)
    cube_in2 = read_cube(in2)
    in2.close()
    assert numpy.allclose(cube_in1["origin"], cube_in2["origin"])
    if op == "sub":
        cube_out = cube_in1["data"] - cube_in2["data"]
    elif op == "add":
        cube_out = cube_in1["data"] + cube_in2["data"]
    elif op == "mul":
        cube_out = cube_in1["data"] * cube_in2["data"]
    elif op == "div":
        cube_out = cube_in1["data"] / cube_in2["data"]
    else:
        raise ValueError(f"operation {op} not supported")
    # write cube file
    out = open(out, "w")
    write_cube(out, cube_in1["atoms"], cube_out, origin=cube_in1["origin"])
    out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="get the difference between two cube files and write to a new cube file"
    )
    parser.add_argument("-f1", "--file1", type=str, help="cube file 1")
    parser.add_argument("-f2", "--file2", type=str, help="cube file 2")
    parser.add_argument("-o", "--output", type=str, help="output cube file name")
    parser.add_argument(
        "-p",
        "--operation",
        type=str,
        required=True,
        choices=["sub", "add", "mul", "div"],
        help="operation to perform: sub, add, mul, div",
    )
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

    cube_operation(
        in1=args.file1,
        in2=args.file2,
        out=args.output,
        op=args.operation
    )
