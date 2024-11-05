import random
import csv
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import copy

@dataclass
class Room:
    id: str
    capacity: int

@dataclass
class Teacher:
    id: str
    subjects: List[str]
    lesson_types: List[str]

@dataclass
class Group:
    id: str
    size: int
    required_subjects: Dict[str, int]  # subject: hours_per_week

@dataclass
class TimeSlot:
    day: int  # 0-4 (Monday to Friday)
    period: int  # 0-3 (4 periods per day)
    
    def __eq__(self, other):
        return self.day == other.day and self.period == other.period

@dataclass
class Lesson:
    subject: str
    teacher: str
    group: str
    room: str
    time_slot: TimeSlot

class Schedule:
    def __init__(self, lessons: List[Lesson]):
        self.lessons = lessons
        self.fitness = None

    def check_hard_constraints(self, rooms: Dict[str, Room], teachers: Dict[str, Teacher], 
                             groups: Dict[str, Group]) -> bool:
        """Check all hard constraints that must be satisfied"""
        
        # Check teacher conflicts (teacher can't be in two places at once)
        teacher_slots = {}
        for lesson in self.lessons:
            key = (lesson.teacher, lesson.time_slot.day, lesson.time_slot.period)
            if key in teacher_slots:
                return False
            teacher_slots[key] = True

        # Check group conflicts (group can't be in two places at once)
        group_slots = {}
        for lesson in self.lessons:
            key = (lesson.group, lesson.time_slot.day, lesson.time_slot.period)
            if key in group_slots:
                return False
            group_slots[key] = True

        # Check room conflicts (room can't be used by two groups at once)
        room_slots = {}
        for lesson in self.lessons:
            key = (lesson.room, lesson.time_slot.day, lesson.time_slot.period)
            if key in room_slots:
                return False
            room_slots[key] = True

        # Check room capacity
        for lesson in self.lessons:
            if rooms[lesson.room].capacity < groups[lesson.group].size:
                return False

        # Check teacher subject compatibility
        for lesson in self.lessons:
            if lesson.subject not in teachers[lesson.teacher].subjects:
                return False

        # Check maximum lessons per day for groups (e.g., max 4 lessons per day)
        for group_id in groups:
            for day in range(5):
                day_lessons = len([l for l in self.lessons 
                                 if l.group == group_id and l.time_slot.day == day])
                if day_lessons > 4:  # Maximum 4 lessons per day
                    return False

        # Check maximum lessons per day for teachers (e.g., max 4 lessons per day)
        for teacher_id in teachers:
            for day in range(5):
                day_lessons = len([l for l in self.lessons 
                                 if l.teacher == teacher_id and l.time_slot.day == day])
                if day_lessons > 4:  # Maximum 4 lessons per day
                    return False

        return True

    def calculate_fitness(self, rooms: Dict[str, Room], teachers: Dict[str, Teacher], 
                         groups: Dict[str, Group], fitness_type: str = "basic") -> float:
        # First check hard constraints
        if not self.check_hard_constraints(rooms, teachers, groups):
            return 0.0

        if fitness_type == "basic":
            return self._calculate_basic_fitness(rooms, teachers, groups)
        else:  # "advanced"
            return self._calculate_advanced_fitness(rooms, teachers, groups)

    def _count_windows(self, sorted_lessons: List[Lesson]) -> int:
        """Count number of empty periods between lessons"""
        windows = 0
        for day in range(5):
            day_lessons = [l for l in sorted_lessons if l.time_slot.day == day]
            if len(day_lessons) < 2:
                continue
            periods = sorted([l.time_slot.period for l in day_lessons])
            for i in range(len(periods) - 1):
                gap = periods[i + 1] - periods[i] - 1
                if gap > 0:
                    windows += gap
        return windows

    def _calculate_basic_fitness(self, rooms: Dict[str, Room], teachers: Dict[str, Teacher], 
                               groups: Dict[str, Group]) -> float:
        total_penalty = 0.0
        
        # Calculate windows in schedule
        for teacher_id in teachers:
            teacher_lessons = sorted(
                [l for l in self.lessons if l.teacher == teacher_id],
                key=lambda x: (x.time_slot.day, x.time_slot.period)
            )
            total_penalty += self._count_windows(teacher_lessons)

        for group_id in groups:
            group_lessons = sorted(
                [l for l in self.lessons if l.group == group_id],
                key=lambda x: (x.time_slot.day, x.time_slot.period)
            )
            total_penalty += self._count_windows(group_lessons)

        # Check if any group has more than 20 lessons per week
        for group_id in groups:
            group_lessons = [l for l in self.lessons if l.group == group_id]
            if len(group_lessons) > 20:
                total_penalty += (len(group_lessons) - 20) * 10

        return 1.0 / (1.0 + total_penalty)

    def _calculate_advanced_fitness(self, rooms: Dict[str, Room], teachers: Dict[str, Teacher], 
                                  groups: Dict[str, Group]) -> float:
        score = 100.0
        
        # Track daily lesson count for groups and teachers
        group_daily_lessons = {group_id: [0] * 5 for group_id in groups}
        teacher_daily_lessons = {teacher_id: [0] * 5 for teacher_id in teachers}
        
        for lesson in self.lessons:
            # Update daily counts
            group_daily_lessons[lesson.group][lesson.time_slot.day] += 1
            teacher_daily_lessons[lesson.teacher][lesson.time_slot.day] += 1
            
            # Penalize for room capacity mismatches
            room_capacity = rooms[lesson.room].capacity
            group_size = groups[lesson.group].size
            capacity_diff = abs(room_capacity - group_size)
            score -= capacity_diff * 0.1
            
            # Penalize afternoon classes
            if lesson.time_slot.period >= 2:
                score -= 0.5
            
            # Reward if the teacher is teaching their primary subject
            teacher_subjects = teachers[lesson.teacher].subjects
            if lesson.subject == teacher_subjects[0]:
                score += 0.5

        # Penalize uneven distribution of lessons across days
        for group_id, daily_counts in group_daily_lessons.items():
            avg_lessons = sum(daily_counts) / 5
            variance = sum((count - avg_lessons) ** 2 for count in daily_counts) / 5
            score -= variance * 2

        # Penalize uneven distribution of teacher workload
        for teacher_id, daily_counts in teacher_daily_lessons.items():
            avg_lessons = sum(daily_counts) / 5
            variance = sum((count - avg_lessons) ** 2 for count in daily_counts) / 5
            score -= variance * 2

        return max(0.0, min(1.0, score / 100))

class ScheduleGenerator:
    def __init__(self, rooms: Dict[str, Room], teachers: Dict[str, Teacher],
                 groups: Dict[str, Group], population_size: int = 50,
                 selection_strategy: str = "greedy",  # "greedy" or "rain"
                 fitness_type: str = "basic"):  # "basic" or "advanced"
        self.rooms = rooms
        self.teachers = teachers
        self.groups = groups
        self.population_size = population_size
        self.selection_strategy = selection_strategy
        self.fitness_type = fitness_type

    def is_valid_lesson_placement(self, lesson: Lesson, existing_lessons: List[Lesson]) -> bool:
        for existing in existing_lessons:
            if lesson.time_slot == existing.time_slot:
                if (lesson.teacher == existing.teacher or 
                    lesson.group == existing.group or 
                    lesson.room == existing.room):
                    return False
        return True

    def generate_valid_schedule(self) -> Schedule:
        lessons = []
        attempts = 0
        max_attempts = 1000

        for group_id, group in self.groups.items():
            for subject, hours in group.required_subjects.items():
                suitable_teachers = [
                    t_id for t_id, t in self.teachers.items()
                    if subject in t.subjects
                ]
                if not suitable_teachers:
                    continue

                for _ in range(hours):
                    valid_placement = False
                    local_attempts = 0
                    
                    while not valid_placement and local_attempts < 100:
                        teacher = random.choice(suitable_teachers)
                        room = random.choice([r_id for r_id, r in self.rooms.items() 
                                           if r.capacity >= group.size])
                        time_slot = TimeSlot(
                            day=random.randint(0, 4),
                            period=random.randint(0, 3)
                        )
                        
                        new_lesson = Lesson(subject, teacher, group_id, room, time_slot)
                        if self.is_valid_lesson_placement(new_lesson, lessons):
                            lessons.append(new_lesson)
                            valid_placement = True
                        
                        local_attempts += 1
                    
                    if not valid_placement:
                        return None

        return Schedule(lessons)

    def generate_initial_population(self) -> List[Schedule]:
        population = []
        attempts = 0
        max_attempts = self.population_size * 10

        while len(population) < self.population_size and attempts < max_attempts:
            schedule = self.generate_valid_schedule()
            if schedule is not None:
                schedule.fitness = schedule.calculate_fitness(self.rooms, self.teachers, self.groups, self.fitness_type)
                if schedule.fitness > 0:
                    population.append(schedule)
            attempts += 1

        return population

    def crossover(self, parent1: Schedule, parent2: Schedule) -> Tuple[Schedule, Schedule]:
        if not parent1.lessons or not parent2.lessons:
            return parent1, parent2

        crossover_point = random.randint(1, len(parent1.lessons)-1)
        
        child1_lessons = parent1.lessons[:crossover_point] + parent2.lessons[crossover_point:]
        child2_lessons = parent2.lessons[:crossover_point] + parent1.lessons[crossover_point:]
        
        return Schedule(child1_lessons), Schedule(child2_lessons)

    def mutate(self, schedule: Schedule, mutation_rate: float = 0.1) -> Schedule:
        """Enhanced mutation strategy with multiple mutation operators"""
        if not schedule:
            return schedule

        new_schedule = copy.deepcopy(schedule)
        
        for lesson in new_schedule.lessons:
            if random.random() < mutation_rate:
                # Choose mutation operator randomly with different weights
                mutation_type = random.choices(
                    ['time', 'room', 'teacher', 'swap', 'shift_day', 'compress'],
                    weights=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
                )[0]
                
                valid_change = False
                attempts = 0
                original_state = copy.deepcopy(lesson)
                
                while not valid_change and attempts < 50:
                    lesson_copy = copy.deepcopy(original_state)
                    other_lessons = [l for l in new_schedule.lessons if l != lesson]
                    
                    if mutation_type == 'time':
                        lesson_copy.time_slot = TimeSlot(
                            day=random.randint(0, 4),
                            period=random.randint(0, 3)
                        )
                    
                    elif mutation_type == 'room':
                        valid_rooms = [r_id for r_id, r in self.rooms.items() 
                                    if r.capacity >= self.groups[lesson.group].size]
                        if valid_rooms:
                            lesson_copy.room = random.choice(valid_rooms)
                    
                    elif mutation_type == 'teacher':
                        suitable_teachers = [
                            t_id for t_id, t in self.teachers.items()
                            if lesson.subject in t.subjects
                        ]
                        if suitable_teachers:
                            lesson_copy.teacher = random.choice(suitable_teachers)
                    
                    elif mutation_type == 'swap':
                        # Find another lesson to swap times with
                        same_subject_lessons = [l for l in other_lessons 
                                             if l.subject == lesson.subject]
                        if same_subject_lessons:
                            swap_with = random.choice(same_subject_lessons)
                            lesson_copy.time_slot, swap_with.time_slot = (
                                swap_with.time_slot, lesson_copy.time_slot
                            )
                    
                    elif mutation_type == 'shift_day':
                        # Shift all lessons for this group to another day
                        shift = random.randint(-2, 2)
                        new_day = (lesson_copy.time_slot.day + shift) % 5
                        lesson_copy.time_slot.day = new_day
                    
                    elif mutation_type == 'compress':
                        # Try to move lesson earlier in the day
                        if lesson_copy.time_slot.period > 0:
                            lesson_copy.time_slot.period -= 1
                    
                    # Check if the change is valid
                    if self.is_valid_lesson_placement(lesson_copy, other_lessons):
                        lesson.time_slot = lesson_copy.time_slot
                        lesson.room = lesson_copy.room
                        lesson.teacher = lesson_copy.teacher
                        valid_change = True
                    
                    attempts += 1
                
                if not valid_change:
                    # Revert to original state if no valid change found
                    lesson.time_slot = original_state.time_slot
                    lesson.room = original_state.room
                    lesson.teacher = original_state.teacher
        
        return new_schedule

    def evolve(self, generations: int = 100) -> Schedule:
        population = self.generate_initial_population()
        
        if not population:
            raise ValueError("Could not generate valid initial population")

        best_ever = None
        generations_without_improvement = 0
        
        for generation in range(generations):
            # Calculate fitness for all schedules
            for schedule in population:
                schedule.fitness = schedule.calculate_fitness(self.rooms, self.teachers, self.groups, self.fitness_type)
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness or 0, reverse=True)
            
            # Update best ever
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best_ever = copy.deepcopy(population[0])
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {population[0].fitness}")
            
            # Early stopping if no improvement for many generations
            if generations_without_improvement > 20:
                print("Early stopping due to no improvement")
                break
            
            # Create new generation through crossover and mutation
            new_population = []
            
            while len(new_population) < self.population_size:
                if len(population) < 2:
                    break
                    
                parent1 = random.choice(population[:len(population)//2])
                parent2 = random.choice(population[:len(population)//2])
                
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Only add valid children with non-zero fitness
                for child in [child1, child2]:
                    if child is not None:
                        fitness = child.calculate_fitness(self.rooms, self.teachers, self.groups)
                        if fitness > 0:
                            child.fitness = fitness
                            new_population.append(child)
            
            # Combine old and new population
            combined_population = population + new_population
            
            # Selection strategy
            if self.selection_strategy == "greedy":
                # Sort by fitness and take the best ones
                combined_population.sort(key=lambda x: x.fitness or 0, reverse=True)
                population = combined_population[:self.population_size]
                
            else:  # "rain" strategy
                # Keep top 10% as elite
                elite_size = 2#max(2, self.population_size // 10)
                combined_population.sort(key=lambda x: x.fitness or 0, reverse=True)
                elite = combined_population[:elite_size]
                
                # Randomly select from the rest
                non_elite = combined_population[elite_size:]
                if non_elite:
                    random_selection = random.sample(
                        non_elite, 
                        min(len(non_elite), self.population_size - elite_size)
                    )
                    population = elite + random_selection
                else:
                    population = elite
            
            if not population:
                raise ValueError("Population became empty during evolution")
        
        return best_ever if best_ever is not None else population[0]

def save_example_data():
    # Save rooms (аудиторії) - increased capacity to accommodate groups
    with open('rooms.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'capacity'])
        writer.writerows([
            ['215', '45'],    # Лекційна аудиторія
            ['321', '35'],    # Комп'ютерний клас
            ['419', '60'],    # Велика лекційна аудиторія
            ['103', '30'],    # Практична аудиторія
            ['205', '50'],    # Велика лекційна аудиторія
            ['316', '35'],    # Лабораторія
        ])

    # Save teachers (викладачі) - broader subject coverage
    with open('teachers.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'subjects', 'lesson_types'])
        writer.writerows([
            ['Петренко О.В.', 'Вища математика,Дискретна математика,Математичний аналіз,Теорія ймовірностей', 'лекція,практика'],
            ['Коваленко І.М.', 'Програмування,Алгоритми та структури даних,Об\'єктно-орієнтоване програмування,Веб-технології', 'лекція,лабораторна,практика'],
            ['Савчук Т.П.', 'Бази даних,Веб-технології,Програмування,Алгоритми та структури даних', 'лекція,лабораторна,практика'],
            ['Мельник В.С.', 'Фізика,Архітектура комп\'ютерів,Комп\'ютерні мережі', 'лекція,лабораторна,практика'],
            ['Іваненко Р.О.', 'Теорія ймовірностей,Математична статистика,Вища математика,Дискретна математика', 'лекція,практика'],
        ])

    # Save groups (групи) - reduced number of groups and subject hours
    with open('groups.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'size', 'subject:hours'])
        writer.writerows([
            ['КН-11', '25', 'Вища математика:3,Програмування:4,Дискретна математика:2,Фізика:2'],
            ['КН-12', '20', 'Вища математика:3,Програмування:4,Бази даних:3,Веб-технології:2'],
            ['ІПЗ-11', '30', 'Об\'єктно-орієнтоване програмування:4,Бази даних:3,Математична статистика:2'],
            ['КБ-11', '15', 'Комп\'ютерні мережі:3,Теорія ймовірностей:2,Алгоритми та структури даних:3'],
        ])

def load_data():
    rooms = {}
    with open('rooms.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rooms[row['id']] = Room(row['id'], int(row['capacity']))

    teachers = {}
    with open('teachers.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            teachers[row['id']] = Teacher(
                row['id'],
                row['subjects'].split(','),
                row['lesson_types'].split(',')
            )

    groups = {}
    with open('groups.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_hours = {}
            for subj_hour in row['subject:hours'].split(','):
                subj, hours = subj_hour.split(':')
                subject_hours[subj] = int(hours)
            groups[row['id']] = Group(row['id'], int(row['size']), subject_hours)

    return rooms, teachers, groups

def save_schedule_to_csv(schedule):
    with open('schedule_output.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['day', 'period', 'group', 'subject', 'teacher', 'room'])
        
        # Sort and write all lessons
        sorted_lessons = sorted(schedule.lessons, 
                              key=lambda x: (x.time_slot.day, x.time_slot.period))
        
        for lesson in sorted_lessons:
            writer.writerow([
                lesson.time_slot.day + 1,  # Day (1-5)
                lesson.time_slot.period + 1,  # Period (1-4)
                lesson.group,
                lesson.subject,
                lesson.teacher,
                lesson.room
            ])

# Modify the main() function to include saving:
def main():
    # Save example data
    save_example_data()
    
    # Load data
    rooms, teachers, groups = load_data()
    
    # Create generator
    generator = ScheduleGenerator(rooms, teachers, groups,    
                                  selection_strategy="rain", 
                                  fitness_type="advanced" )
    
    try:
        # Generate schedule
        best_schedule = generator.evolve(generations=100)
        
        # Print results
        print("\nBest schedule found:")
        print(f"Fitness: {best_schedule.fitness}")
        
        # Save schedule to CSV
        save_schedule_to_csv(best_schedule)
        print("\nSchedule has been saved to 'schedule_output.csv'")
        
        # Print schedule by day (keeping the existing console output)
        for day in range(5):
            print(f"\nDay {day+1}:")
            day_lessons = [l for l in best_schedule.lessons if l.time_slot.day == day]
            day_lessons.sort(key=lambda x: x.time_slot.period)
            
            for lesson in day_lessons:
                print(f"Period {lesson.time_slot.period+1}: "
                      f"Group {lesson.group} - {lesson.subject} "
                      f"(Teacher: {lesson.teacher}, Room: {lesson.room})")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
