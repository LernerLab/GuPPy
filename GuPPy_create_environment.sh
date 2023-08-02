#!/bin/bash

# Check if the OS is Linux
if [[ "$(uname)" == "Linux" ]]; then
	echo "Linux operating system detected."
	sudo apt-get update
	apt-get install wget
	wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
	sha256sum Anaconda3-2022.05-Linux-x86_64.sh
	bash Anaconda3-2022.05-Linux-x86_64.sh
	rm -rf Anaconda3-2022.05-Linux-x86_64.sh
	conda_file="./GuPPy_environment_ubuntu.yml"
  
# Check if the OS is macOS
elif [[ "$(uname)" == "Darwin" ]]; then
	echo "macOS operating system detected."
	curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-MacOSX-x86_64.sh
	sha256sum Anaconda3-2022.05-MacOSX-x86_64.sh
	bash Anaconda3-2022.05-MacOSX-x86_64.sh
	rm -rf Anaconda3-2022.05-MacOSX-x86_64.sh
	# Fetch the Conda YAML file from the specified path
	conda_file="./GuPPy_environment_mac.yml"

# Check if the OS is Windows (using MSYS or Cygwin)
elif [[ "$(uname -o 2>/dev/null)" == "Cygwin" || "$(uname -o 2>/dev/null)" == "Msys" ]]; then
	echo "Windows operating system detected."

# If none of the above, assume it's an unknown OS
else
	echo "$(uname)"
	echo "Unknown operating system detected."
fi

# Check if the Conda YAML file exists
if [ ! -f "$conda_file" ]; then
	echo "Conda YAML file not found in the specified path."
	exit 1
fi

# Extract the environment name from the Conda YAML file
env_name=$(grep -e 'name: ' "$conda_file" | awk '{print $2}')

# Check if the Conda environment with the same name exists
if conda env list | grep -q -w "$env_name"; then
	echo "Conda environment '$env_name' already exists. Removing it..."
	conda remove -n "$env_name" --all
fi

# Create a new Conda environment using the fetched YAML file
conda env create -f "$conda_file"

# activate conda environment
source ./GuPPy_activate_environment.sh

echo "Conda environment '$env_name' created successfully."
