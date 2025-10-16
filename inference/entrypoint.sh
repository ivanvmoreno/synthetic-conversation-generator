#!/bin/bash

# Install Python dependencies from requirements.txt
pip install -r /app/requirements.txt

# Execute the provided CMD
exec "$@"