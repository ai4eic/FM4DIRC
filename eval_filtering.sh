#!/bin/bash

for p in 3.0 6.0 9.0
do
  echo "Running DLL for ${p} GeV/c"
  python eval_filtering.py --config config/CA_config.json --momentum $p 
  echo "------------------------------------------------- "
  echo " " 
done