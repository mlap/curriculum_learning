#!/bin/bash
make_plots(){
  for R in 0.01 0.1 0.3 0.5 1.0
  do
  	for K in 0.5 1.0 2.0
  	do
  		python eval_fishing_agent.py --model $1 --r $R --K $K
    done
  done
}
