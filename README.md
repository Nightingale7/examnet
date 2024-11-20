# Contents
This repository contains the code for the Examnet algorithm as well as the code used for running the experiments on the arch-com 2024 and ambiegen benchmarks.

# Installation Requirements
The algorithm is built on top of the STGEM testing tool ( https://gitlab.abo.fi/aidoart/stgem ), which needs to be installed.
* Follow the instructions in the INSTALLATION.md file of the STGEM to install the tool.
* After STGEM is installed, clone this repository into the same folder as STGEM.

# Running the code
* The arch-com 2024 benchmarks can be executed by running the command: `./run_arch_experiments_2024.sh`. 
* The ambiegen benchmarks can be executed with `./run_ambiegen_experiments_2024.sh`
* Output for the experiments can be found in `./outputs` folder and contains the number of executions required to falsify the benchmark. 

