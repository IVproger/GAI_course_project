#!/usr/bin/env bash

# Adds the current working directory to the PYTHONPATH
echo 'export PYTHONPATH="$PYTHONPATH:$(pwd)"' >> ~/.bashrc
source ~/.bashrc