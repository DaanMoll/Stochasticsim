#!/bin/bash
echo Linear
python3 markov.py Linear

echo Log
python3 markov.py Log

echo Quadratic
python3 markov.py Quadratic

echo Exponential
python3 markov.py Exponential