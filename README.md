# Multi-Objective Optimization of a PWR Secondary Steam Cycle

This repository contains the Python implementation of the multi-objective optimization framework for a Pressurized Water Reactor (PWR) secondary steam cycle, as described in the paper: *"[Insert Your Paper Title Here]"*.

## Overview
The project uses the `pymoo` framework to execute the Non-Dominated Sorting Genetic Algorithm II (NSGA-II) to optimize the reheat pressure and steam extraction pressures of a 12-component steam cycle. Shannon Entropy and VIKOR are employed to determine objective weights and isolate the optimal compromise solution.

## Prerequisites
To run these scripts, you will need Python 3.8+ and the following dependencies:
```bash
pip install numpy pymoo pandas matplotlib
