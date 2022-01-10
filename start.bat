@echo off
start cmd /k "uvicorn api:app --reload --port=8008"