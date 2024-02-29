"""
convert xyz to aims' geometry.in
"""
import argparse
import os
from argparse import Namespace

try:
    from tqdm import tqdm

    iterate = lambda x: tqdm(enumerate(x), total=len(x))
except ImportError:
    print("tqdm not found, progress bar disabled")
    iterate = enumerate

from ase.io import read, write


def main(args: Namespace):
    assert os.path.exists(args.input), "input file not found"
    assert os.path.exists(args.output), "output dir not found"

    xyzfile = read(args.input, index=":")

    if args.all:
        print(f"converting all structures in xyz file, cnt = {len(xyzfile)}")
        idx_len = len(str(len(xyzfile)))  # for zero padding
        for i, xyz in iterate(xyzfile):
            if args.one:
                j = i + 1
            out_fpath = os.path.join(args.output, "geometry.in.{}".format(str(j)))
            write(out_fpath, xyz, format="aims")
            assert os.path.exists(out_fpath)
    else:
        print(f"converting the {args.num}-th structure in xyz file")
        # NOTE: args.num should minus args.one!
        if args.one:
            num = int(args.num) - 1
            print(f"number starts from 1")
        else:
            num = int(args.num)
            print(f"number starts from 0")
        assert num < len(xyzfile), "structure number out of range"
        out_fpath = os.path.join(args.output, "geometry.in")
        write(out_fpath, xyzfile[num], format="aims")
        assert os.path.exists(out_fpath)

    print("done, and verified all output files exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="file path to input xyz file")
    parser.add_argument(
        "-o",
        "--output",
        help="dir path to output aims geometry.in files, \
        should be an existing dir",
    )
    parser.add_argument("-n", "--num", help="convert the n-th structure in xyz file")
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="convert all structures in xyz file, \
        this will override the -n option, and output filenames will be appended with the structure number. \
        numbers are without zero padding",
    )
    parser.add_argument(
        "-1",
        "--one",
        action="store_true",
        help="number starts from 1, \
        default is 0",
    )
    args = parser.parse_args()

    main(args)
