# MOPP-DEC via SAT

This project implements the SAT formulation of the MOPP-DEC decision problem
as described in the accompanying paper.

## Structure
- Models/: problem instance definitions
- CaseStudies/: example problem instances
- Solvers/: SAT solver implementation
- main.py: run a case study

## Install
```bash
pip install python-sat[pblib,aiger]
```
## Run
```bash
python main.py
```