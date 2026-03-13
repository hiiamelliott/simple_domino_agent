#!/bin/bash
# Domino App launcher — starts the chat UI on port 8888.
# Use this as the entry point when creating a Domino App.

PORT=8888
python chat_app.py --debug --port $PORT
