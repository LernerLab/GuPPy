#!/bin/bash

source ./GuPPy_activate_environment.sh

file_path="./../guppy.log"
if [ -f "$file_path" ]; then
	echo "Previous log file exists. Deleting..."
	rm -rf "$file_path"
	echo "Previous log file deleted."
else
	echo "Previous log file does not exist."
fi

read -p "Do you want to change the folder path in GuPPy's Input Parameters GUI (default foldr path is $HOME)? Proceed(y/n?) : " user_input

user_input_lower=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')

# Check the user input
if [[ "$user_input_lower" == "y" ]]; then
	# Check if 'zenity' is installed
	if command -v zenity >/dev/null 2>&1; then
		# Launch the directory selector dialog and store the selected directory path
		selected_dir=$(zenity --file-selection --directory --title "Select a directory")

		# Check if a directory was selected
		if [ -n "$selected_dir" ]; then
			echo "Folder path changed to $selected_dir"
		else
			echo "Folder path not set"
			selected_dir=$HOME
			echo "Folder path set to $selected_dir by default"
		fi
	else
		echo "'zenity' is not installed. Installing zenity"
		if [[ "$(uname)" == "Linux" ]]; then
			echo "Linux operating system detected."
		elif [[ "$(uname)" == "Darwin" ]]; then
			brew install zenity
		elif [[ "$(uname -o 2>/dev/null)" == "Cygwin" || "$(uname -o 2>/dev/null)" == "Msys" ]]; then
			echo "Windows operating system detected."
		else
			echo "Unknown operating system detected."
		fi

		# Launch the directory selector dialog and store the selected directory path
		selected_dir=$(zenity --file-selection --directory --title "Select a path to the folder")

		# Check if a directory was selected
		if [ -n "$selected_dir" ]; then
			echo "Folder path set to $selected_dir"
		else
			echo "Folder path not set"
			selected_dir=$HOME
			echo "Folder path set to $selected_dir by default"
		fi
	fi

elif [[ "$user_input_lower" == "n" ]]; then
	selected_dir=$HOME
	if [ -n "$selected_dir" ]; then
		echo "Folder path set to $selected_dir"
	else
		echo "Folder path not set"
	fi

else
	echo "Invalid input: '$user_input'. Only 'y' or 'n' allowed."
	exit 1
fi

guppy_path=$(pwd)
panel serve --show ./GuPPy/savingInputParameters.ipynb --args "$selected_dir" "$guppy_path" 