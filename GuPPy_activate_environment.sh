#!/bin/bash

# Check if the OS is Linux
if [[ "$(uname)" == "Linux" ]]; then
	source "$(conda info --base)/bin/activate" guppy
  
# Check if the OS is macOS
elif [[ "$(uname)" == "Darwin" ]]; then
	echo "$(uname)"
	source "$(conda info --base)/bin/activate" guppy

# Check if the OS is Windows (using MSYS or Cygwin)
elif [[ "$(uname -o 2>/dev/null)" == "Cygwin" || "$(uname -o 2>/dev/null)" == "Msys" ]]; then
	source "$(conda info --base)/Scripts/activate" guppy

# If none of the above, assume it's an unknown OS
else
	echo "$(uname)"
	echo "Unknown operating system detected."
fi