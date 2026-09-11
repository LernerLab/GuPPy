# Installation

GuPPy runs on Windows, macOS and Linux, and requires **Python 3.10 or greater**.

## Step 1: Install Conda

We recommend installing GuPPy inside a conda virtual environment to avoid conflicts with other Python packages on your system.

1. Download the **Miniconda** installer for your operating system from the [official Miniconda page](https://docs.conda.io/en/latest/miniconda.html).

   - **Windows**: download the `.exe` installer and run it, following the on-screen prompts.
   - **macOS**: download the `.pkg` installer (or the `.sh` script) and follow the on-screen prompts. Alternatively, run the shell script in a terminal:

     ```bash
     bash Miniconda3-latest-MacOSX-x86_64.sh
     ```

   - **Linux**: download the `.sh` installer and run it in a terminal:

     ```bash
     bash Miniconda3-latest-Linux-x86_64.sh
     ```

2. After installation, open a new terminal (or Command Prompt / Anaconda Prompt on Windows) and verify conda is available:

   ```bash
   conda --version
   ```

## Step 2: Create and activate a conda environment

1. Create a new conda environment named `guppy_env` with Python 3.12:

   ```bash
   conda create -n guppy_env python=3.12
   ```

2. Activate the environment:

   ```bash
   conda activate guppy_env
   ```

   Your terminal prompt should now show `(guppy_env)` to indicate the environment is active. You will need to activate this environment each time you open a new terminal before using GuPPy.

## Step 3: Install GuPPy

With the `guppy_env` environment active, install GuPPy using one of the two methods below.

### Option A: Install from PyPI (recommended)

```bash
pip install guppy-neuro
```

```{note}
GuPPy 2.0 is in beta. No 2.0 stable release has been published yet, so this installs the latest
beta. If you need a stable version, see the [v1.3.0 release](https://github.com/LernerLab/GuPPy/releases/tag/v1.3.0),
which predates the 2.0 rewrite and is documented on the [GitHub Wiki](https://github.com/LernerLab/GuPPy/wiki).
```

### Option B: Install from GitHub (latest development version)

This option gives you access to the latest features and bug fixes that may not yet be in the stable release. You will need `git` installed ([installation instructions](https://github.com/git-guides/install-git)).

1. Clone the repository:

   ```bash
   git clone https://github.com/LernerLab/GuPPy.git
   ```

2. Navigate into the cloned directory:

   ```bash
   cd GuPPy
   ```

3. Install the package in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs):

   ```bash
   pip install -e .
   ```

## Launching GuPPy

In a terminal with `guppy_env` active, run:

```bash
guppy
```

This launches the GuPPy user interface, where you can begin analyzing your fiber photometry data. The [first analysis tutorial](tutorials/first_analysis.md) walks through a complete session from there.

## Updating GuPPy

Check which version you have with `guppy --version`. It is also shown in the user interface header and recorded as `guppy_version` in every `GuPPyParamtersUsed.json`; quote it when reporting a problem.

If you installed from PyPI (Option A):

```bash
pip install --upgrade guppy-neuro
```

If you installed from GitHub (Option B), pull the latest source and reinstall from your clone:

```bash
git pull
pip install -e .
```

## Sample data included in the repository

The first tutorial analyzes a small CSV session stored in the repository under `stubbed_testing_data/`. Those files are tracked with [Git LFS](https://git-lfs.com), so a plain clone leaves you with pointer files rather than the recordings themselves. If you installed from source and want the sample session, install Git LFS once per machine and fetch the files:

```bash
git lfs install
git lfs pull --include="stubbed_testing_data/csv/sample_data_csv_1/*"
```

## Setting up a development environment

Contributors who need the test, docs or linting dependencies should follow the [development environment guide](contributing/development_environment.md), which covers GuPPy's [PEP 735](https://peps.python.org/pep-0735/) dependency groups.
