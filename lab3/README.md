# University Schedule Generator

## Overview
A genetic algorithm implementation for university schedule generation, tailored for the Ukrainian education system. The system handles complex scheduling constraints and provides multiple optimization strategies.

## Features
- CSV-based data management for easy modification
- Multiple selection strategies:
 - Greedy approach (sort by fitness and select best)
 - Rain selection method (keep top 10% elite + random selection)
- Two quality assessment functions:
 - Basic: focuses on windows minimization and basic constraints
 - Advanced: considers room optimization, teacher specialization and workload balance
- Advanced mutation system with 6 different operators:
 - Time slot change (30%)
 - Room change (20%)
 - Teacher change (20%)
 - Time slot swap (10%)
 - Day shift (10%)
 - Schedule compression (10%)
- Hard constraints:
 - No teacher conflicts
 - No group conflicts 
 - No room conflicts
 - Room capacity check
 - Maximum 20 lessons per week

## Usage
```python
# Initialize generator with desired strategy
generator = ScheduleGenerator(
    rooms, 
    teachers, 
    groups, 
    selection_strategy="rain",  # or "greedy"
    fitness_type="advanced"     # or "basic"
)

# Generate schedule
best_schedule = generator.evolve(generations=100)

# Save to CSV
save_schedule_to_csv(best_schedule)
```
