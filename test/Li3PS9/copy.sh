#!/bin/bash

# param 1: choose method
# param 2: DFT(B) path
scale=(0 1 2 3 4 5 6 7 8 9 10)


# copy detailed.out from folder DFT_std_tight
if  [ $1 -eq 1 ]; then
  for ii in ${scale[@]}; do
    cp $2/aims.out.$ii  ./S$ii/DFT/aims.out
    echo $ii
  done
fi

# copy detailed.out from folder DFTB_std
if  [ $1 -eq 2 ]; then
  for ii in ${scale[@]}; do
    cp $2/detailed.out.$ii  ./S$ii/DFTB/detailed.out
    echo $ii
  done
fi

# copy detailed.out from folder DFT_std_tight
if  [ $1 -eq 3 ]; then
  for ii in ${scale[@]}; do
    cp ./S$ii/DFT/aims.out $2/aims.out.$ii
    echo $ii
  done
fi


# copy detailed.out from folder DFTB_std
if  [ $1 -eq 4 ]; then
  for ii in ${scale[@]}; do
    cp ./S$ii/DFTB/detailed.out $2/detailed.out.$ii
    echo $ii
  done
fi

