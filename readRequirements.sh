#!/bin/bash

for line in $(cat requirements.txt)
do
    pip install $line
done