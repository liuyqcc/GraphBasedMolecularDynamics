# Graph-based Molecular Dynamics (GMD) method
Source code and example input files of the Graph-based Molecular Dynamics (GMD) method.
The code is still under development. Use at your own risk.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- xTB (https://github.com/grimme-lab/xtb)
- MNDO (https://mndo.kofo.mpg.de/)


## Example 1
1. Navigate to the directory: `cd examples/1_GMD_butadiene_S0`
2. Run GMD simulation: `python3 -u path-to-code/run_GMD_butadiene_S0.py`

## Example 2
1. Navigate to the directory: `cd examples/2_GMD_fulvene_S0`
2. Run GMD simulation: `python3 -u path-to-code/run_GMD_fulvene_S0.py`

## Example 3
1. Set the active space in line 87 of the `GraphBasedMolecularDynamics.py` file.
2. Navigate to the directory: `cd examples/3_GMD_styrene_S1`
3. Run GMD simulation: `python3 -u path-to-code/run_GMD_styrene_S1.py`

## Example 4
1. Set the active space in line 87 of the `GraphBasedMolecularDynamics.py` file.
2. Navigate to the directory: `cd examples/4_multistate_reaction_network_S1`
3. Run multistate reaction network search: `python3 -u path-to-code/run_multistate_reaction_network_S1.py`

## Example 5
1. Navigate to the directory: `cd examples/5_multistate_reaction_network_T1`
2. Run multistate reaction network search: `python3 -u path-to-code/run_multistate_reaction_network_T1.py`