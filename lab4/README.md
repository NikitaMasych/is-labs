# Constraint Satisfaction Problem (CSP) Solver

This project implements a solver for **Constraint Satisfaction Problems (CSPs)** using backtracking search enhanced with heuristics. CSPs are defined by:
- A set of **variables**.
- **Domains** of possible values for each variable.
- **Constraints** that specify valid combinations of variable assignments.

## Features
- **Heuristics for Efficiency:**
  - **Minimum Remaining Values (MRV):** Selects the variable with the smallest domain first.
  - **Least Constraining Value (LCV):** Prioritizes values that minimize conflicts with other variables.
- **Input/Output Format:**
  - Input data (domains and constraints) are read from CSV files.
  - Solution is saved in a human-readable CSV file.

## How It Works
1. **Input Files:**
   - `domains.csv`: Defines the variables and their possible values.
   - `constraints.csv`: Defines the constraints between variables.

2. **Algorithm:**
   - Backtracking search is used to assign values to variables while respecting all constraints.
   - Heuristics improve efficiency by reducing conflicts and minimizing search depth.

3. **Output File:**
   - `solution.csv`: Contains the valid assignments for all variables.

## Example Usage
- Input:
  - `domains.csv`:
    ```
    variable,value
    G1,T1-R1-TS1
    G1,T2-R2-TS2
    G2,T1-R1-TS2
    G2,T2-R2-TS3
    ```
  - `constraints.csv`:
    ```
    variables,condition
    G1,G2,G1.split("-")[2] != G2.split("-")[2]
    G1,G3,G1.split("-")[1] != G3.split("-")[1]
    ```
- Output:
  - `solution.csv`:
    ```
    variable,assigned_value
    G1,T1-R1-TS1
    G2,T2-R2-TS3
    ```

## Requirements
- Python 3.x
- CSV file handling libraries (default in Python)

## Run the Program
1. Clone the repository.
2. Add your input files (`domains.csv`, `constraints.csv`) to the project directory.
3. Run the main script:
   ```bash
   python csp_solver.py
4. Check the solution in solution.csv.
