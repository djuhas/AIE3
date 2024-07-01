#!/bin/bash

# Get the absolute path of the repository root
repo_root=$(git rev-parse --show-toplevel)

# Ensure script is run from the repository root
cd "$repo_root"

# Stash local changes to allow branch switching
git stash -u

# Add the AI-Maker-Space repository as a remote if it does not exist
if ! git remote | grep -q "ai-maker-space"; then
  git remote add ai-maker-space https://github.com/AI-Maker-Space/AI-Engineering-3.git
fi

# Fetch the latest changes from the AI-Maker-Space repository
git fetch ai-maker-space

# Create a new branch to track the changes from AI-Maker-Space
git checkout -b temp-branch ai-maker-space/main

# Identify new files and directories
new_files=$(git diff --name-status main..temp-branch | grep "^A" | cut -f2-)

# Debug: Print the new files and directories identified
echo "New files and directories:"
echo "$new_files"

# Initialize an empty array for successfully copied files
copied_files=()

# Copy new files and directories to the main branch directory if they don't exist
IFS=$'\n' # Handle file names with spaces correctly
for file in $new_files; do
  # Check if the file already exists in the destination
  if [ ! -e "$repo_root/$file" ]; then
    # Debug: Print each file being copied
    echo "Copying $file"
  
    # Create the directory structure if it does not exist
    mkdir -p "$(dirname "$repo_root/$file")"
    # Copy the file or directory
    cp -r "$file" "$repo_root/$file"
    # Add to copied files array
    copied_files+=("$file")
  else
    # Debug: Print a message if the file already exists
    echo "File $file already exists, skipping."
  fi
done

# Verify if the files were copied
echo "Verifying copied files:"
for file in "${copied_files[@]}"; do
  if [ -f "$repo_root/$file" ] || [ -d "$repo_root/$file" ]; then
    echo "$file successfully copied."
  else
    echo "Failed to copy $file."
  fi
done

# Switch back to the main branch
git checkout main

# Apply stashed changes
git stash pop

# Add and commit the new files, excluding the script itself
git add .
git reset update_from_ai_maker_space.sh
git commit -m "Added new files from AI-Engineering-3 repository"

# Delete the temporary branch
git branch -d temp-branch

# Push the changes to the remote repository if needed
echo "Your branch is ahead of 'origin/main' by $(git rev-list --count origin/main..main) commit(s)."
echo "To publish your local commits, use 'git push'."
