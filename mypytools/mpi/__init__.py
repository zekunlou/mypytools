def distribute_work(job_indexes: list[int], size: int, random: bool = True) -> list[list[int]]:
    """split the job_indexes into size parts"""
    import random

    if random:
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
    assert len(job_indexes_per_task) == size, (
        f"the number of tasks should be equal to {size}, but got {len(job_indexes_per_task)=}"
    )
    return job_indexes_per_task
