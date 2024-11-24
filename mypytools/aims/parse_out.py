import re

import numpy

"""
strings to match: (always take the last one)
```text
  | Number of atoms                   :       28
  | Electrostatic energy          :       -1980.39868384 Ha      -53889.39004924 eV
  | XC energy correction          :        -151.50354247 Ha       -4122.62114728 eV
  | Electronic free energy        :       -1067.13189040 Ha      -29038.13618179 eV
```
sometimes it will be like
```text
  | Electronic free energy        :  -443732128.61910671 Ha ******************** eV
```
"""


def parse_aims_out(*args, **kwargs):
    version = kwargs.get("version", 241124)
    if "version" in kwargs:
        kwargs.pop("version")
    if version == 241124:
        return parse_aims_out_241124(*args, **kwargs)


_regexp_dict = {  # energy in eV
    "number_of_atoms": re.compile(r"\|\s*Number of atoms\s*:\s*(\d+)"),
    "electrostatic_energy": re.compile(r"\|\s*Electrostatic energy\s*:\s*-?\d+\.\d+\s*Ha\s*(-?\d+\.\d+)\s*eV"),
    "xc_energy": re.compile(r"\|\s*XC energy correction\s*:\s*-?\d+\.\d+\s*Ha\s*(-?\d+\.\d+)\s*eV"),
    "total_energy": re.compile(r"\|\s*Electronic free energy\s*:\s*-?\d+\.\d+\s*Ha\s*(-?\d+\.\d+)\s*eV"),
}


def parse_aims_out_241124(fpath: str):
    """
    this function is used in viper salted/2406ppr
    match all patterns in regexp_dict, and take the last one
    if not found, return numpy.nan
    """
    with open(fpath) as f:
        content = f.read()
    results = {}
    for key, regexp in _regexp_dict.items():
        m = regexp.findall(content)
        if m:
            if "number" in key:
                results[key] = int(m[-1])
            elif "energy" in key:
                results[key] = float(m[-1])
            else:
                results[key] = float(m[-1])
        else:
            print(f"{key=} not found at {fpath=}")
            results[key] = numpy.nan
    return results


# parse_aims_out("/u/zklou/projects/salted/2406ppr/graphene/cal_ksdft/inter_pred_bandstr/1/aims.out")
# parse_aims_out("/u/zklou/projects/salted/2406ppr/graphene/inter_3x3/optim_soap_r3/aims_predicted_data/1/aims.out")
