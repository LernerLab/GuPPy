[![DOI](https://zenodo.org/badge/382176345.svg)](https://zenodo.org/badge/latestdoi/382176345) [![Join the chat at https://gitter.im/LernerLab/GuPPy](https://badges.gitter.im/LernerLab/GuPPy.svg)](https://gitter.im/LernerLab/GuPPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
# GuPPy
 Guided Photometry Analysis in Python, a free and open-source fiber photometry data analysis tool.

## Installation

GuPPy can be run on Windows, Mac or Linux.

### Installation via PyPI

To install the latest stable release of GuPPy through PyPI, simply run the following command in your terminal or command prompt:

```bash
pip install guppy
```

We recommend that you install the package inside a [virtual environment](https://docs.python.org/3/tutorial/venv.html). 
A simple way of doing this is to use a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) from the `conda` package manager ([installation instructions](https://docs.conda.io/en/latest/miniconda.html)). 
Detailed instructions on how to use conda environments can be found in their [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Installation from GitHub

To install the latest development version of GuPPy from GitHub, you can clone the repository and install the package manually. 
This has the advantage of allowing you to access the latest features and bug fixes that may not yet be available in the stable release. 
To install the conversion from GitHub you will need to use `git` ([installation instructions](https://github.com/git-guides/install-git)). 
From a terminal or command prompt, execute the following commands:

1. Clone the repository:
```bash
git clone https://github.com/LernerLab/GuPPy.git
```

2. Navigate into the cloned directory:
```bash
cd GuPPy
```

3. Install the package using pip:
```bash
pip install -e .
```

Note:
This method installs the repository in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs).

## Usage

In a terminal or command prompt, you can start using GuPPy by running the following command:

```bash
guppy
```

This will launch the GuPPy user interface, where you can begin analyzing your fiber photometry data.

## Wiki
- The full instructions along with detailed descriptions of each step to run the GuPPy tool is on [Github Wiki Page](https://github.com/LernerLab/GuPPy/wiki).

## Tutorial Videos

- [Installation steps](https://youtu.be/7qfU8xvj2nc)
- [Explaining Input Parameters GUI](https://youtu.be/aO7_QqbYZ84)
- [Individual Analysis steps](https://youtu.be/6IollIr9q6Y)
- [Artifacts Removal](https://youtu.be/KXh3vkkZxuo)
- [Group Analysis steps](https://youtu.be/lntf-SER_so)
- [Use of csv file as an input](https://youtu.be/Yrhartn5Hwk)
- [Use of Neurophotometrics data as an input](https://youtu.be/n1HSGRnBYPQ)

## Sample Data

- [Sample data](https://drive.google.com/drive/folders/1qO8ynfqRoEpWuJ0P1tYVHtLljJXoxufl?usp=sharing) for the user to go through the tool in the start. This folder of sample data has two types of sample data recorded with a TDT system : 1) Clean Data 2) Data with artifacts (to practice removing them) 3) Neurophotometrics data 4) Doric system data. Finally, it has a control channel, signal channel and event timestamps file in a 'csv' format to get an idea of how to structure other data in the 'csv' file format accepted by GuPPy.

## Discussions

- GuPPy was initially developed keeping our data (FP data recorded using TDT systems) in mind. GuPPy now supports data collected using Neurophotometrics, Doric system and also other data types/formats using 'csv' files as input, but these are less extensively tested because of lack of sample data. If you have any issues, please get in touch on the [chat room](https://gitter.im/LernerLab/GuPPy?utm_source=share-link&utm_medium=link&utm_campaign=share-link) or by [raising an issue](https://github.com/LernerLab/GuPPy/issues), so that we can continue to improve this tool.

## Citation

- If you use GuPPy for your research, please cite [Venus N. Sherathiya, Michael D. Schaid, Jillian L. Seiler, Gabriela C. Lopez, and Talia N. Lerner GuPPy, a Python toolbox for the analysis of fiber photometry data](https://www.nature.com/articles/s41598-021-03626-9)

> Venus N. Sherathiya, Michael D. Schaid, Jillian L. Seiler, Gabriela C. Lopez, and Talia N. Lerner GuPPy, a Python toolbox for the analysis of fiber photometry data. Sci Rep 11, 24212 (2021). https://doi.org/10.1038/s41598-021-03626-9

## Contributors

- [Venus Sherathiya](https://github.com/venus-sherathiya)
- [Michael Schaid](https://github.com/Mschaid)
- Jillian Seiler
- [Gabriela Lopez](https://github.com/glopez924)
- [Talia Lerner](https://github.com/talialerner)
- [Paul Adkisson](https://github.com/pauladkisson)


