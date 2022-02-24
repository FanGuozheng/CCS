#!/bin/bash

scale=(0 1 2 3 4 5 6 7 8 9 10)

for ii in ${scale[@]}; do
  mkdir -p S$ii/DFT
  mkdir -p S$ii/DFTB
  echo $ii
done


