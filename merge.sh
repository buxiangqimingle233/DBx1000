#!/bin/bash

# Prompt user for source and destination directories
read -p "Enter source directory: " src_dir
read -p "Enter destination directory: " dst_dir

# Extract the patterns from directory names
src_pattern=$(echo "$src_dir" | grep -oE '^[0-9-]+')
dst_pattern=$(echo "$dst_dir" | grep -oE '^[0-9-]+')

# Check if source directory exists
if [ ! -d "$src_dir" ]; then
    echo "Source directory does not exist. Exiting."
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$dst_dir"

# Function to rename and move files and directories
rename_and_move() {
    local current_dir="$1"
    local target_dir="$2"

    for file in "$current_dir"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, create a corresponding directory in the destination
            # and recursively call this function
            local new_dir_name=$(basename "$file" | sed "s/$src_pattern/$dst_pattern/")
            local new_dir="$target_dir/$new_dir_name"
            mkdir -p "$new_dir"
            rename_and_move "$file" "$new_dir"
        else
            # If it's a file, move and rename it
            mv "$file" "$target_dir/$(basename "$file" | sed "s/$src_pattern/$dst_pattern/")"
        fi
    done
}

# Call the function with the source and destination directories
rename_and_move "$src_dir" "$dst_dir"

echo "Operation completed."
