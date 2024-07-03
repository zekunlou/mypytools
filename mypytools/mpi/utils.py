""" do not import mpi4py in the header! """
import os
import random
from typing import List

def load_print_func(use_mpi:bool=True) -> callable:
    if not use_mpi:
        return print
    else:
        _, _, rank = load_mpi(use_mpi)
        return lambda *args, **kwargs: print(f"{rank=}:", *args, **kwargs)

def load_mpi(use_mpi:bool=True):
    """load MPI objects, and keep consistency between MPI and none MPI

    Args:
        use_mpi: if use mpi

    Returns: tuple (comm, size, rank)
    """
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        comm = None
        size = 1
        rank = 0
    return comm, size, rank

def set_num_cpus(num: int = None):
    if isinstance(num, int):
        assert num > 0
        for env_var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            os.environ[env_var] = num

def distribute_work(job_indexes: List[int], size: int, shuffle: bool = True) -> List[List[int]]:
    """split the job_indexes into size parts

    Example:
    ```python
    size = 6
    all_task_indexes = list(range(20))
    if rank == 0:
        my_task_indexes:List[List[int]] = distribute_work(all_task_indexes, size)
    else:
        my_task_indexes = None
    my_task_indexes: List[int] = comm.scatter(my_task_indexes, root=0)
    ```
    """
    if shuffle:
        random.shuffle(job_indexes)
    assert size >= 1, f"size should be greater than or equal to 1, but got {size=}"
    if size == 1:
        return [job_indexes]
    if size > len(job_indexes):
        raise ValueError(
            f"the number of tasks should be less than the number of jobs, but got {size=}, {len(job_indexes)=}"
        )
    n_jobs = len(job_indexes)
    n_jobs_per_task = n_jobs // size
    n_jobs_left = n_jobs % size
    job_indexes_per_task = []
    start_idx = 0
    for i in range(size):
        end_idx = start_idx + n_jobs_per_task
        if i < n_jobs_left:
            end_idx += 1
        job_indexes_per_task.append(job_indexes[start_idx:end_idx])
        start_idx = end_idx
    assert (
        len(job_indexes_per_task) == size
    ), f"the number of tasks should be equal to {size}, but got {len(job_indexes_per_task)=}"
    return job_indexes_per_task
