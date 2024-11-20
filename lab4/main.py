import csv
from typing import List, Dict, Any, Tuple


class CSP:
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], constraints: List[Tuple[List[str], str]]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.assignments = {}

    def is_consistent(self, var, value):
        """
        Check if assigning a value to a variable is consistent with the constraints.
        """
        for (vars_, condition) in self.constraints:
            if var in vars_:
                local_context = {v: self.assignments[v] for v in vars_ if v in self.assignments}
                local_context[var] = value
                if len(local_context) == len(vars_):  # All variables in the constraint are assigned
                    if not eval(condition, {}, local_context):
                        return False
        return True

    def select_unassigned_variable(self):
        """
        Use MRV heuristic to select the next variable to assign.
        """
        unassigned_vars = [v for v in self.variables if v not in self.assignments]
        return min(unassigned_vars, key=lambda var: len([v for v in self.domains[var] if self.is_consistent(var, v)]))

    def order_domain_values(self, var):
        """
        Use Least Constraining Value heuristic to order domain values.
        """
        def count_conflicts(value):
            conflicts = 0
            for (vars_, _) in self.constraints:
                if var in vars_:
                    for neighbor in vars_:
                        if neighbor != var and neighbor not in self.assignments:
                            for neighbor_value in self.domains[neighbor]:
                                if not self.is_consistent(neighbor, neighbor_value):
                                    conflicts += 1
            return conflicts

        return sorted(self.domains[var], key=lambda val: count_conflicts(val))

    def backtrack(self):
        """
        Perform backtracking search to solve the CSP.
        """
        if len(self.assignments) == len(self.variables):
            return self.assignments

        var = self.select_unassigned_variable()

        for value in self.order_domain_values(var):
            if self.is_consistent(var, value):
                self.assignments[var] = value
                result = self.backtrack()
                if result is not None:
                    return result
                del self.assignments[var]

        return None


def save_example_data():
    """
    Save a slightly simpler example for CSP in the same format as Lab 3.
    """
    # Save variables and domains
    with open('domains.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['variable', 'value'])
        writer.writerows([
            # Group G1
            ['G1', 'T1-R1-TS1'], ['G1', 'T2-R2-TS2'],
            # Group G2
            ['G2', 'T1-R1-TS2'], ['G2', 'T2-R2-TS3'],
            # Group G3
            ['G3', 'T1-R2-TS1'], ['G3', 'T2-R3-TS2'],
            # Group G4
            ['G4', 'T1-R3-TS4'], ['G4', 'T2-R1-TS1'],
        ])

    # Save constraints
    with open('constraints.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['variables', 'condition'])
        writer.writerows([
            # Constraints for time slots: no overlapping time slots
            ['G1,G2', 'G1.split("-")[2] != G2.split("-")[2]'],
            ['G1,G3', 'G1.split("-")[2] != G3.split("-")[2]'],
            ['G2,G4', 'G2.split("-")[2] != G4.split("-")[2]'],
            # Constraints for rooms: no overlapping rooms
            ['G1,G3', 'G1.split("-")[1] != G3.split("-")[1]'],
            ['G2,G4', 'G2.split("-")[1] != G4.split("-")[1]'],
            # Constraints for teachers: no teacher teaches two groups at the same time
            ['G1,G2', 'G1.split("-")[0] != G2.split("-")[0]'],
            ['G3,G4', 'G3.split("-")[0] != G4.split("-")[0]'],
        ])




def load_data():
    """
    Load variables, domains, and constraints from CSV files.
    """
    variables = []
    domains = {}
    constraints = []

    # Load domains
    with open('domains.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row['variable']
            if var not in variables:
                variables.append(var)
            if var not in domains:
                domains[var] = []
            domains[var].append(row['value'])

    # Load constraints
    with open('constraints.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vars_ = row['variables'].split(',')
            condition = row['condition']
            constraints.append((vars_, condition))

    return variables, domains, constraints


def save_solution(assignments: Dict[str, str]):
    """
    Save the solution in the same format as Lab 3.
    """
    with open('solution.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['variable', 'assigned_value'])
        for var, value in assignments.items():
            writer.writerow([var, value])


def main():
    # Save example data
    save_example_data()

    # Load data
    variables, domains, constraints = load_data()
    print(variables, domains, constraints)

    # Solve CSP
    csp = CSP(variables, domains, constraints)
    solution = csp.backtrack()

    if solution:
        print("Solution found:")
        for var, value in solution.items():
            print(f"{var} -> {value}")
        save_solution(solution)
        print("\nSolution saved to 'solution.csv'")
    else:
        print("No solution exists.")


if __name__ == "__main__":
    main()
