#!/bin/bash

# Ensure script is run from the repository root
cd "$(git rev-parse --show-toplevel)"

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

# Copy new files and directories to the main branch directory
IFS=$'\n' # Handle file names with spaces correctly
for file in $new_files; do
  # Debug: Print each file being copied
  echo "Copying $file"
  
  # Create the directory structure if it does not exist
  mkdir -p "$(dirname "../$file")"
  # Copy the file or directory
  cp -r "$file" "../$file"
done

# Switch back to the main branch
git checkout main

# Apply stashed changes
git stash pop

# Add and commit the new files
git add .
git commit -m "Added new files from AI-Engineering-3 repository"

# Delete the temporary branch
git branch -d temp-branch
