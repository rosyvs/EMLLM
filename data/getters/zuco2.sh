#!/bin/bash

# Define the OSF project ID
PROJECT_ID="2urht"

# Local download path is command line argument. Parse this even with a space in the path
DOWNLOAD_PATH="$1"

# go to the download path
cd "$DOWNLOAD_PATH"

# List all files in the project
osf -p $PROJECT_ID ls > file_list.txt

# Read the file_list.txt and download files except those with "Raw data" in the path
while IFS= read -r line
do
    if [[ $line != *"Raw data"* ]]; then
        # make folder structure
        mkdir -p "$(dirname "$line")"
        osf -p $PROJECT_ID fetch "$line" "$line"
        fi
    done < file_list.txt

