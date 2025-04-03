import os

import numpy


def move_ri_output(
    dpath: str,
    ovlp_input: str = "ri_ovlp.out",
    ovlp_output: str = "overlap.npy",
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
                assert n**2 == data.size
                data = data.reshape((n, n))
            numpy.save(os.path.join(dpath, fname_output), data)
