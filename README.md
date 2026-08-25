[![DOI](https://zenodo.org/badge/382176345.svg)](https://zenodo.org/badge/latestdoi/382176345) [![Join the chat at https://gitter.im/LernerLab/GuPPy](https://badges.gitter.im/LernerLab/GuPPy.svg)](https://gitter.im/LernerLab/GuPPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![codecov](https://codecov.io/gh/LernerLab/GuPPy/graph/badge.svg)](https://codecov.io/gh/LernerLab/GuPPy) [![Documentation Status](https://readthedocs.org/projects/guppy/badge/?version=latest)](https://guppy.readthedocs.io/)

<img src="https://raw.githubusercontent.com/LernerLab/GuPPy/main/assets/GuppyLogo.png" alt="GuPPy Logo" width="300">

Guided Photometry Analysis in Python, a free and open-source fiber photometry data analysis tool.

> **GuPPy 2.0 is in beta.** `pip install guppy-neuro` installs a pre-release. If you need a stable version, see [Older versions](#older-versions).

## Quick start

Requires Python 3.10 or greater. We recommend installing into a conda environment:

```bash
conda create -n guppy_env python=3.12
conda activate guppy_env
pip install guppy-neuro
```

Then launch the user interface:

```bash
guppy
```

See the [installation guide](https://guppy.readthedocs.io/en/latest/installation.html) for the full walkthrough, including installing conda and installing from source.

## Documentation

The GuPPy documentation lives at [guppy.readthedocs.io](https://guppy.readthedocs.io/): a [tutorial](https://guppy.readthedocs.io/en/latest/tutorials/index.html) walking through a first analysis end to end, [how-to guides](https://guppy.readthedocs.io/en/latest/how-to/index.html) for individual tasks, [explanations](https://guppy.readthedocs.io/en/latest/explanation/index.html) of how the analysis works, and a [reference](https://guppy.readthedocs.io/en/latest/reference/index.html) for every input parameter and output file.

## Citation

- If you use GuPPy for your research, please cite [Venus N. Sherathiya, Michael D. Schaid, Jillian L. Seiler, Gabriela C. Lopez, and Talia N. Lerner GuPPy, a Python toolbox for the analysis of fiber photometry data](https://www.nature.com/articles/s41598-021-03626-9)

> Venus N. Sherathiya, Michael D. Schaid, Jillian L. Seiler, Gabriela C. Lopez, and Talia N. Lerner GuPPy, a Python toolbox for the analysis of fiber photometry data. Sci Rep 11, 24212 (2021). https://doi.org/10.1038/s41598-021-03626-9

## Contributors

- [Venus Sherathiya](https://github.com/venus-sherathiya)
- [Michael Schaid](https://github.com/Mschaid)
- Jillian Seiler
- [Gabriela Lopez](https://github.com/glopez924)
- [Talia Lerner](https://github.com/talialerner)
- [Paul Adkisson-Floro](https://github.com/pauladkisson)

## Older versions

No GuPPy 2.0 stable release has been published yet. If you need a stable version, use **[GuPPy v1.3.0](https://github.com/LernerLab/GuPPy/releases/tag/v1.3.0)**. It predates the 2.0 rewrite and is not installable with `pip`: download the source from that release, then from inside the unpacked folder run

```bash
# Substitute spec_file_windows10.txt or spec_file_linux.txt for your OS.
conda create --name guppy --file spec_file_mac.txt
conda activate guppy
panel serve --show GuPPy/savingInputParameters.ipynb
```

v1.3.0 has a different interface and different parameters, so the documentation at guppy.readthedocs.io does not apply to it. Use the [GitHub Wiki](https://github.com/LernerLab/GuPPy/wiki) instead.
