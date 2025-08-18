#!/bin/bash
# Build the Docker image
docker build -t pytabular .

# Run the Streamlit app
docker run --rm -p 8501:8501 pytabular
