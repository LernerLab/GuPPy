# GuPPy
 Guided Photometry Analysis in Python, a free and open-source FP analysis tool.

## Installation Instructions

GuPPy can be run on Windows, Mac or Linux

Follow the instructions below to install GuPPy

- Download the Guppy code
- Install [Anaconda](https://www.anaconda.com/products/individual#macos). Install Anaconda based on your operating system (Mac, Windows or Linux)
- Open Anaconda Prompt window (for windows) or open terminal window (for Mac or Linux)
- Go to the location where GuPPy folder is located using the following command
```
cd path_to_GuPPy_folder
```
- Execute the following commands on Anaconda Prompt or terminal window <br>
Note : filename in the first command should be replaced by <b>spec_file_windows10.txt</b> or <b>spec_file_mac.txt</b> or <b>spec_file_linux.txt</b> (based on your OS)
```
conda create --name guppy --file filename
conda activate guppy
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=guppy
```
