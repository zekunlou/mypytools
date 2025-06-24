import os

import numpy


def move_ri_output(
    dpath: str,
    ovlp_input: str = "ri_ovlp.out",
    ovlp_output: str = "overlap.npy",  # you can put absolute path here
    proj_input: str = "ri_projections.out",
    proj_output: str = "projections.npy",
    coef_input: str = "ri_restart_coeffs.out",
    coef_output: str = "coefficients.npy",
    allow_skip: bool = True,
):
    if not os.path.exists(dpath):
        raise FileNotFoundError(f"{dpath} does not exist")
    for fname_input in (ovlp_input, proj_input, coef_input):
        fpath_input = os.path.join(dpath, fname_input)
        if not os.path.exists(fpath_input):
            raise FileNotFoundError(f"{fpath_input} does not exist")
    for data_name, (fname_output, fname_input) in (
        (
            "ovlp",
            (ovlp_output, ovlp_input),
        ),
        (
            "proj",
            (proj_output, proj_input),
        ),
        (
            "coef",
            (coef_output, coef_input),
        ),
    ):
        if os.path.exists(os.path.join(dpath, fname_output)) and allow_skip:
            continue
        else:
            data = numpy.loadtxt(os.path.join(dpath, fname_input))
            if data_name == "ovlp":
                n = int(numpy.round(data.size**0.5))
                assert n**2 == data.size, f"Overlap matrix size mismatch: {data.size} is not a perfect square"
                data = data.reshape((n, n))
            numpy.save(os.path.join(dpath, fname_output), data)


def main() -> None:
    """
    Main function to parallelize SALTED data conversion.

    This function:
    1. Sets up MPI communication
    2. Identifies all job indices
    3. Distributes jobs among processes
    4. Each process handles its assigned jobs
    5. Reports timing statistics
    """
    import argparse
    import time

    from mypytools.mpi.utils import distribute_work, load_mpi, load_print_func, set_num_cpus

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="Use MPI for parallel processing",
    )
    args = parser.parse_args()
    use_mpi = args.mpi

    # Initialize MPI
    comm, size, rank = load_mpi(use_mpi)
    my_print = load_print_func(use_mpi)

    # Set number of threads per process to 1 to avoid oversubscription
    set_num_cpus(1)

    # Root process identifies all jobs and distributes them
    current_dpath = os.getcwd()

    # Only rank 0 checks directory structure and collects job indices
    if rank == 0:
        # Verify directory structure
        subdir_names = os.listdir(current_dpath)
        required_subdirs = ["data", "coefficients", "overlaps", "projections"]

        for subdname in required_subdirs:
            assert subdname in subdir_names, f"{subdname} not found in {current_dpath}"

        # Set paths
        data_dpath = os.path.join(current_dpath, "data")
        ovlp_dpath = os.path.join(current_dpath, "overlaps")
        proj_dpath = os.path.join(current_dpath, "projections")
        coef_dpath = os.path.join(current_dpath, "coefficients")

        # Find all AIMS indices (directories that are numbers)
        aims_indices = [
            int(dname)
            for dname in os.listdir(data_dpath)
            if os.path.isdir(os.path.join(data_dpath, dname)) and dname.isdigit()
        ]

        my_print(f"Found {len(aims_indices)} AIMS indices to process")

        # Distribute work among processes
        distributed_aims_indices = distribute_work(aims_indices, size, shuffle=True)
    else:
        # Non-root processes initialize these variables
        data_dpath = None
        ovlp_dpath = None
        proj_dpath = None
        coef_dpath = None
        distributed_aims_indices = None

    # Broadcast paths to all processes
    if use_mpi:
        data_dpath = comm.bcast(data_dpath, root=0)
        ovlp_dpath = comm.bcast(ovlp_dpath, root=0)
        proj_dpath = comm.bcast(proj_dpath, root=0)
        coef_dpath = comm.bcast(coef_dpath, root=0)
        # Scatter the work to each process
        my_aims_indices = comm.scatter(distributed_aims_indices, root=0)
    else:
        my_aims_indices = distributed_aims_indices[0]  # Only one process in non-MPI mode

    # Process the assigned jobs
    my_print(f"Processing {len(my_aims_indices)} jobs")

    total_time = 0.0
    for i, aims_idx in enumerate(my_aims_indices):
        salted_idx = aims_idx - 1
        aims_idx_str = str(aims_idx)
        salted_idx_str = str(salted_idx)

        time_start = time.time()

        try:
            move_ri_output(
                dpath=os.path.join(data_dpath, aims_idx_str),
                ovlp_output=os.path.join(ovlp_dpath, f"overlap_conf{salted_idx_str}.npy"),
                proj_output=os.path.join(proj_dpath, f"projections_conf{salted_idx_str}.npy"),
                coef_output=os.path.join(coef_dpath, f"coefficients_conf{salted_idx_str}.npy"),
            )

            time_end = time.time()
            elapsed_time = time_end - time_start
            total_time += elapsed_time

            # Progress reporting
            if (i + 1) % max(1, len(my_aims_indices) // 10) == 0 or i == len(my_aims_indices) - 1:
                my_print(
                    f"Progress: {i + 1}/{len(my_aims_indices)}, Converted {aims_idx_str} in {elapsed_time:.2f} seconds"
                )

        except Exception as e:
            my_print(f"Error processing {aims_idx_str}: {str(e)}")

    # Gather timing information from all processes
    if use_mpi:
        all_times = comm.gather(total_time, root=0)
        all_job_counts = comm.gather(len(my_aims_indices), root=0)

        if rank == 0:
            total_jobs = sum(all_job_counts)
            avg_time_per_job = sum(all_times) / total_jobs if total_jobs > 0 else 0
            max_process_time = max(all_times)

            my_print(f"All processes completed.")
            my_print(f"Total jobs processed: {total_jobs}")
            my_print(f"Average time per job: {avg_time_per_job:.4f} seconds")
            my_print(f"Max process time: {max_process_time:.2f} seconds")
            my_print(f"Estimated speedup: {total_jobs * avg_time_per_job / max_process_time:.2f}x")
    else:
        my_print(f"Completed {len(my_aims_indices)} jobs in {total_time:.2f} seconds")
        my_print(f"Average time per job: {total_time / len(my_aims_indices):.4f} seconds")


if __name__ == "__main__":
    main()
