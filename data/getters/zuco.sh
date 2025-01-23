#!/bin/bash

# Define the OSF project ID
PROJECT_ID="q3zws"

# Local download path is command line argument. Parse this even with a space in the path
DOWNLOAD_PATH="$1"

# go to the download path
cd "$DOWNLOAD_PATH"

# check if file_list.txt exists, otherwise create it and list all files in the project
if [ ! -f file_list.txt ]; then
    osf -p $PROJECT_ID ls > file_list.txt
fi

# Read the file_list.txt and download files except those with "Raw data" in the path
while IFS= read -r line
do
    if [[ $line != *"Raw data"* ]]; then
        # if the file does not exist, download it
        if [ ! -f "$line" ]; then
            # make folder structure
            mkdir -p "$(dirname "$line")"
            osf -p $PROJECT_ID fetch "$line" "$line"
            fi
        fi
    done < file_list.txt
