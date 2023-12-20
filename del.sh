#!/bin/bash

# Define an array of user-installed packages
user_installed_packages=(
    "beautifulsoup4"
    "docker"
    "docker-compose"
    "dockerpty"
    "matplotlib"
    "numpy"
    "scipy"
    "pythran"
    "sympy"
    "yfinance"
)

# Update package lists
sudo apt-get update

# Loop through the array and remove each package
for pkg in "${user_installed_packages[@]}"; do
    echo "Removing $pkg..."
    sudo apt-get purge -y "$pkg"
done

# Clean up
sudo apt-get autoremove -y
sudo apt-get autoclean -y

echo "Package removal complete."

