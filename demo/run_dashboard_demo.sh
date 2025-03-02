#!/bin/bash

# Create assets directory if it doesn't exist
mkdir -p /home/cabdru/marketing/demo/assets

# Create a symbolic link to the logo.png file
ln -sf /home/cabdru/marketing/demo/demo_assets/logo.png /home/cabdru/marketing/demo/assets/logo.png

# Run the dashboard demo
python3 /home/cabdru/marketing/demo/demo_dashboard.py