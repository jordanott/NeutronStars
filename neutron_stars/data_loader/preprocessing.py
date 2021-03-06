"""
This script parses *.dat.gz files
---
Each .dat file is structured like this:

# Star 0
 # Spectral Coefficients: n, lambda(1), ..., lambda(n)
 2    5.004855178267951E+00   -1.950356165346860E+00
# Mass\tRadius\tnH\tlogTeff\tdist
1.2047833E+00\t1.2255060E+01\t0.104526\t6.049572\t6.086809
# Spectrum
0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0....

The parsed data is saved as a NumPy file like:
{
    star_nums : (N, 1),
    details   : (N, 5),
    coefficients : (N, [2/4]),
    spectra      : (N, 1024),
}
"""

import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import iglob


def parse_file(file_name):

    with gzip.open(file_name, 'r') as f:
        for i, line in enumerate(f):
            pass

        idx = 0
        num_lines = i // 7
        num_coefficients = 4 if '4Param' in file_name else 2
        star_nums = np.zeros((num_lines, 1), dtype=np.int)
        details = np.zeros((num_lines, 5), dtype=np.float)
        coefficients = np.zeros((num_lines, num_coefficients), dtype=np.float)
        spectra = np.zeros((num_lines, 1024), dtype=np.int)

        f.seek(0)
        f.readline()
        while True:
            star_num = f.readline().decode('UTF-8')
            if not star_num: break
            star_num = int(star_num.replace('# Star', ''))
            _ = f.readline()

            coefficient = f.readline().decode('UTF-8')
            coefficient = list(map(float, coefficient.split('   ')))
            _ = f.readline()

            detail = f.readline().decode('UTF-8')
            detail = list(map(float, detail.split('\t')))
            _ = f.readline()

            spectrum = f.readline().decode('UTF-8')
            spectrum = list(map(int, spectrum.split(', ')[:-1]))

            star_nums[idx] = star_num
            details[idx] = detail
            coefficients[idx] = coefficient[1:]
            spectra[idx] = spectrum
            idx += 1

    np.savez(
        file_name.replace('.dat.gz', '.npz'),
        star_nums=star_nums,
        details=details,
        coefficients=coefficients,
        spectra=spectra,
    )


if __name__ == '__main__':
    # tar -xvzf /baldig/physicstest/NeutronStarsData/res.tgz
    all_files = list(iglob(DATA_DIR + '*.dat.gz'))

    for file_name in tqdm(all_files):
        parse_file(file_name)
