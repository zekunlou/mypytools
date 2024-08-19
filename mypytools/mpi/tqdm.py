import io
import sys
import time
from typing import Literal, Sequence, Union

_time_formatter = {
    "mmss": lambda x: time.strftime("%M:%S", time.gmtime(x)),
    "hhmmss": lambda x: time.strftime("%H:%M:%S", time.gmtime(x)),
    "ddhhmmss": lambda x: time.strftime("%d-%H:%M:%S", time.gmtime(x)),
}


class TqdmBar:
    def __init__(
        self,
        desc: str,
        total: int,
        ncols: int,
        flush: bool = True,
        file: io.TextIOWrapper = sys.stderr,
    ):
        self.desc = desc
        self.total = total
        self.ncols = ncols  # not used
        self.file = file
        self.flush = flush
        self.done = 0
        self.start_time = time.time()
        self.update_time_last = time.time()
        self.update_time_this = None
        self.items_time = []  # tuple (items, time cost)

    def get_speed(
        self,
        unit: Union[Literal["it/s"], Literal["s/it"]] = None,
    ):
        """get the averaged speed of progress

        Returns: Tuple
            speed: float, larger than 1
            unit: str, "it/s" or "s/it"
        """
        # assume s/it
        if len(self.items_time) == 0:
            return 0.0, "it/s"  # Avoid division by zero
        avg_speed = [n / t for n, t in self.items_time if t > 0]
        avg_speed = sum(avg_speed) / len(avg_speed)
        if unit is None:
            if avg_speed > 1:
                return avg_speed, "it/s"
            else:
                return 1 / avg_speed, "s/it"
        elif unit == "it/s":
            return avg_speed, "it/s"
        elif unit == "s/it":
            return 1 / avg_speed, "s/it"
        else:
            raise ValueError(f"unit should be 'it/s' or 's/it', got {unit}")

    def time_formatter(self, seconds: float):
        if seconds < 3600:
            time_formatter = _time_formatter["mmss"]
        elif seconds < 86400:
            time_formatter = _time_formatter["hhmmss"]
        else:
            time_formatter = _time_formatter["ddhhmmss"]
        return time_formatter(seconds)

    def compose_str(
        self,
    ):
        frac_progress = f"{self.done}/{self.total}"
        percentage = self.done / self.total * 100
        percentage = f"{percentage:3f}%"
        run_time = time.time() - self.start_time
        # format as dd-hh:mm:ss or hh:mm:ss or mm:ss
        time_cost = self.time_formatter(run_time)
        if self.done == 0:
            time_togo = self.time_formatter(0)
        else:
            time_togo = self.time_formatter(run_time * (self.total - self.done) / self.done)
        speed, unit = self.get_speed()
        if self.desc is None:
            desc_str = ""
        else:
            desc_str = self.desc + ": "
        return f"{desc_str}{percentage}, {frac_progress} [{time_cost}<{time_togo}, {speed:.2f}{unit}]"

    @property
    def finished(self):
        return self.done >= self.total

    def update(self, n=1):
        self.update_time_this = time.time()
        self.done += n
        self.items_time.append((n, self.update_time_this - self.update_time_last))
        self.update_time_last = self.update_time_this

    def refresh(self):
        refresh_str = self.compose_str()
        print("\r" + refresh_str, end="", file=self.file, flush=self.flush)

    def close(self):
        print("", file=self.file, flush=self.flush)  # just new line


class tqdmMPI:
    """tqdm for MPI parallelization, support both Sequence and update method

    Usage:
    ```python
    all_tasks:List[int] = list(range(100))
    my_tasks:List[int] = distribute_work(all_tasks, size)
    for data in tqdmMPI(my_tasks):
        ...
    ```
    ```python
    my_tasks_total_size:int = 100
    pbar = tqdmMPI(total=my_tasks_total_size)
    for _ in range(my_tasks_total_size):
        ...
        pbar.update()
    ```
    """

    UPDATE_TAG = 11415

    def __init__(
        self,
        iterable: Sequence = None,  # distributed jobs for each rank, don't use iterator!
        desc: str = None,
        total: int = None,  # jobs for this rank
        ncols: int = 80,
        file: io.TextIOWrapper = sys.stderr,
        mpi: bool = True,
    ):
        if mpi:
            from mpi4py import MPI

            self.mpi_comm = MPI.COMM_WORLD
            self.mpi_size = self.mpi_comm.Get_size()
            self.mpi_rank = self.mpi_comm.Get_rank()
            self.mpi_any_source = MPI.ANY_SOURCE
        else:
            self.mpi_comm = None
            self.mpi_size = 1
            self.mpi_rank = 0
            self.mpi_any_source = None
        self.desc = desc
        if iterable is not None:
            assert total is None, "total should not be provided when iterable is provided"
            self.iterable = iterable
            self.total = len(self.iterable)
        elif iterable is None:
            assert total is not None, "total should be provided when iterable is not provided"
            self.iterable = None
            self.total = total
        self.total_sum = self.mpi_comm.allreduce(self.total, op=MPI.SUM)
        self.ncols = ncols  # not used
        self.file = file

        self.prog_bar = None
        if self.mpi_rank == 0:
            self.prog_bar = TqdmBar(self.desc, self.total_sum, self.ncols, file=self.file)
            self.prog_bar.refresh()

    def __iter__(self):
        if self.iterable is not None:
            for item in self.iterable:
                yield item
                self._check_for_updates()
                self.update()
        else:
            for _ in range(self.total):
                yield
                self._check_for_updates()
                self.update()
        if self.mpi_rank == 0:  # continue to update until all ranks finish
            while not self.prog_bar.finished:
                self._check_for_updates()
                self.update()
                time.sleep(self.prog_bar.get_speed("s/it")[0])

    def update(self, n=1):
        """all ranks send n to rank 0 in nonblocking way, rank 0 update the progress bar"""
        if self.mpi_comm is not None:
            self.mpi_comm.send(n, dest=0, tag=self.UPDATE_TAG)
            if self.mpi_rank == 0:
                for _ in range(self.mpi_size):
                    # n = self.mpi_comm.irecv(source=self.any_source, tag=self.UPDATE_TAG)
                    req = self.mpi_comm.irecv(source=self.mpi_any_source, tag=self.UPDATE_TAG)
                    n = req.wait()
                    self.prog_bar.update(n)
                self.prog_bar.refresh()
        else:
            self.prog_bar.update(n)
            self.prog_bar.refresh()

    def _check_for_updates(self):
        if self.mpi_comm is not None and self.mpi_rank == 0:
            while True:
                flag = self.mpi_comm.Iprobe(source=self.mpi_any_source, tag=self.UPDATE_TAG)
                if not flag:
                    break
                req = self.mpi_comm.irecv(source=self.mpi_any_source, tag=self.UPDATE_TAG)
                n = req.wait()
                self.prog_bar.update(n)
                self.prog_bar.refresh()


def _test_tqdmMPI():
    import random

    data = list(range(100))
    for _ in tqdmMPI(data):
        time.sleep(random.randrange(1, 10) / 10)
    for _ in tqdmMPI(total=100):
        time.sleep(random.randrange(1, 10) / 10)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_tqdmMPI", action="store_true", help="test class tqdmMPI")
    parser.add_argument("--mpi", action="store_true", help="use mpi or not")
    args = parser.parse_args()
    print(f"{dict(args)=}")

    if args.test_tqdmMPI:
        _test_tqdmMPI(args.mpi)
