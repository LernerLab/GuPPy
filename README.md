[![DOI](https://zenodo.org/badge/382176345.svg)](https://zenodo.org/badge/latestdoi/382176345) [![Join the chat at https://gitter.im/LernerLab/GuPPy](https://badges.gitter.im/LernerLab/GuPPy.svg)](https://gitter.im/LernerLab/GuPPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
# GuPPy
 Guided Photometry Analysis in Python, a free and open-source FP analysis tool.

## Installation Instructions

GuPPy can be run on Windows, Mac or Linux.

**Follow the instructions below to install GuPPy :**

1. Download the Guppy code <br>
   a. Press the green button labeled “Code” on the top right corner and that will initiate a pull down menu. <br>
   
   b. Click on Download ZIP. (*Mac Users: do not save to the Cloud, see note below) <br>
   
   c. Once downloaded, open ZIP file and you should have a folder named “GuPPy-main”. Place this GuPPy-main folder wherever is most convenient. <br>
   
   d. Inside the GuPPy-main folder there is a subfolder named “GuPPy”. Take note of the Guppy subfolder location or path. It will be important for future steps in the GuPPy workflow <br>
   - Mac: Right click folder → Click Get Info → Text next to “Where:” <br>
       ~ Ex: /Users/LernerLab/Desktop/GuPPy-main <br>
   - Windows/Linux: Right click folder → Properties → Text next to “Location:” <br>

2. Install [Anaconda](https://www.anaconda.com/products/individual#macos). Install Anaconda based on your operating system (Mac, Windows or Linux).

3. Once installed, open an Anaconda Prompt window (for windows) or Terminal window (for Mac or Linux).

4. Find the location where GuPPy folder is located (from Step 1d) and execute the following command on the Anaconda Prompt or terminal window: 

```
cd path_to_GuPPy_folder
```
   - Ex:  cd /Users/LernerLab/Desktop/GuPPy-main
  
5. Next, execute the following commands, in this specific order, on Anaconda Prompt or terminal window: <br>
   - Note : filename in the first command should be replaced by <b>spec_file_windows10.txt</b> or <b>spec_file_mac.txt</b> or <b>spec_file_linux.txt</b> (based on your OS) <br>
   - Some of these commands will initiate various transactions. Wait until they are all done before executing the next line <br>
   - If the Anaconda Prompt or Terminal window asks: Proceed ([y]/n)? Respond with y <br>
```
conda create --name guppy --file filename
conda activate guppy
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=guppy
```
6. Open the GuPPy-main folder and click into the GuPPy subfolder. In this subfolder, there will be a file named [savingInputParameters.ipynb](https://github.com/LernerLab/GuPPy/blob/main/GuPPy/savingInputParameters.ipynb) Identify the path/location of this file as similarly described in Step 1d. 

7. On the terminal window or Anaconda prompt window, use the savingInputParameters.ipynb path/location to execute the following command: 

```
cd path_to_file
```
 - Ex: cd /Users/LernerLab/Desktop/GuPPy-main/GuPPy
 
8. Lastly, execute the following command to open the GuPPy User Interface:
```
panel serve --show savingInputParameters.ipynb
```
<b> GuPPy is now officially downloaded and ready to use! <b>

- The full instructions along with detailed descriptions of each step to run the GuPPy tool is on [Github Wiki Page](https://github.com/LernerLab/GuPPy/wiki).

## Tutorial Videos

- [Installation steps](https://youtu.be/7qfU8xvj2nc)
- [Explaining Input Parameters GUI](https://youtu.be/A3wfkG4n7J8)
- [Individual Analysis steps](https://youtu.be/6IollIr9q6Y)
- [Artifacts Removal](https://youtu.be/OKiRZZxKj6Y)
- [Group Analysis steps](https://youtu.be/9JYt5k1zguw)
- [Use of csv file as an input](https://youtu.be/1_hec_GV2_A)

## Sample Data

- [Sample data](https://northwestern.box.com/s/pmzpqey540gkftka669frax84ofk5f4h) for the user to go through the tool in the start. It has two sample data : 1) Clean Data 2) Data with artifacts. It also has a control channel, signal channel and event timestamps file in  'csv' format to get an idea about the structure of the 'csv' file.

## Discussions

- GuPPy is at a very early stage, and was developed with our data (FP data recorded using TDT systems) in mind. In future developments, we hope to support other data types/formats. If you have any issues, please get in touch on the [chat room](https://gitter.im/LernerLab/GuPPy?utm_source=share-link&utm_medium=link&utm_campaign=share-link) or by [raising an issue](https://github.com/LernerLab/GuPPy/issues).

## Contributors

- [Venus Sherathiya](https://github.com/venus-sherathiya)
- Michael Schaid
- Jillian Seiler
- Gabriela Lopez
- [Talia Lerner](https://github.com/talialerner)


