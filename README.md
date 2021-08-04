[![DOI](https://zenodo.org/badge/382176345.svg)](https://zenodo.org/badge/latestdoi/382176345) [![Join the chat at https://gitter.im/LernerLab/GuPPy](https://badges.gitter.im/LernerLab/GuPPy.svg)](https://gitter.im/LernerLab/GuPPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
# GuPPy
 Guided Photometry Analysis in Python, a free and open-source FP analysis tool.

## Installation Instructions

GuPPy can be run on Windows, Mac or Linux.

Follow the instructions below to install GuPPy : 

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
- On the terminal window or Anaconda prompt window, go to the folder where [savingInputParameters.ipynb](https://github.com/LernerLab/GuPPy/blob/main/GuPPy/savingInputParameters.ipynb) file is located using the folowing command. 
```
cd path_to_the_file
```
- Execute the following command to open GuPPy.
```
panel serve --show savingInputParameters.ipynb
```
- Whenever user opens a new terminal window for running the GuPPy code. Please execute the following command
```
conda activate guppy
```
- All the basic instructions along with detailed instructions of each step to run the GuPPy tool is on [Github Wiki Page](https://github.com/LernerLab/GuPPy/wiki).

## Sample Data

- [Sample data](https://northwestern.box.com/s/pmzpqey540gkftka669frax84ofk5f4h) for the user to go through the tool in the start. It has two sample data : 1) Clean Data 2) Data with artifacts. It also has a event timestamps file in  'csv' format to get an idea about the structure of the 'csv' file.

## Discussions

- GuPPy is at a very early stage, and was developed with our data (FP data recorded using TDT systems) in mind. In future developments, we hope to support other data types/formats. If you have any issues, please get in touch on the [chat room](https://gitter.im/LernerLab/GuPPy?utm_source=share-link&utm_medium=link&utm_campaign=share-link) or by [raising an issue](https://github.com/LernerLab/GuPPy/issues).

## Contributors

- [Venus Sherathiya](https://github.com/venus-sherathiya)
- Michael Schaid
- Jillian Seiler
- Gabriela Lopez
- [Talia Lerner](https://github.com/talialerner)


