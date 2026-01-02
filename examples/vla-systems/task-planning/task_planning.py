#!/usr/bin/env python3
"""
Task Planning Example for Educational Robotics

This example demonstrates hierarchical task planning for educational robotics,
showing how to break down complex educational activities into executable steps.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

class TaskStatus(Enum):
    """Status of a task in the planning system"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    """Type of task for educational robotics"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    SAFETY = "safety"

@dataclass
class Task:
    """Represents a single task in the educational robotics system"""
    id: str
    name: str
    task_type: TaskType
    description: str
    prerequisites: List[str] = field(default_factory=list)
    duration: float = 1.0  # Estimated duration in seconds
    priority: int = 1  # Higher number means higher priority
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    children: List['Task'] = field(default_factory=list)
    parent: Optional['Task'] = None

    def execute(self) -> bool:
        """Execute the task and return success status"""
        print(f"Executing task: {self.name} (ID: {self.id})")

        # Simulate task execution based on type
        if self.task_type == TaskType.NAVIGATION:
            return self.execute_navigation()
        elif self.task_type == TaskType.MANIPULATION:
            return self.execute_manipulation()
        elif self.task_type == TaskType.PERCEPTION:
            return self.execute_perception()
        elif self.task_type == TaskType.COMMUNICATION:
            return self.execute_communication()
        elif self.task_type == TaskType.LEARNING:
            return self.execute_learning()
        elif self.task_type == TaskType.SAFETY:
            return self.execute_safety()
        else:
            print(f"Unknown task type: {self.task_type}")
            return False

    def execute_navigation(self) -> bool:
        """Execute navigation task"""
        destination = self.parameters.get('destination', 'unknown')
        speed = self.parameters.get('speed', 0.5)

        print(f"  Navigating to {destination} at speed {speed}")
        time.sleep(min(self.duration, 2.0))  # Simulate navigation
        print(f"  Reached destination: {destination}")
        return True

    def execute_manipulation(self) -> bool:
        """Execute manipulation task"""
        object_name = self.parameters.get('object', 'unknown')
        action = self.parameters.get('action', 'grasp')

        print(f"  Manipulating {object_name} with action {action}")
        time.sleep(min(self.duration, 1.5))  # Simulate manipulation
        print(f"  Completed manipulation of {object_name}")
        return True

    def execute_perception(self) -> bool:
        """Execute perception task"""
        target = self.parameters.get('target', 'unknown')

        print(f"  Perceiving {target}")
        time.sleep(min(self.duration, 1.0))  # Simulate perception
        print(f"  Completed perception of {target}")
        return True

    def execute_communication(self) -> bool:
        """Execute communication task"""
        message = self.parameters.get('message', 'Hello')

        print(f"  Communicating: {message}")
        time.sleep(min(self.duration, 0.5))  # Simulate communication
        print(f"  Message delivered: {message}")
        return True

    def execute_learning(self) -> bool:
        """Execute learning task"""
        topic = self.parameters.get('topic', 'general')

        print(f"  Teaching about {topic}")
        time.sleep(min(self.duration, 2.0))  # Simulate teaching
        print(f"  Completed teaching about {topic}")
        return True

    def execute_safety(self) -> bool:
        """Execute safety task"""
        print("  Performing safety check")
        time.sleep(min(self.duration, 0.5))  # Simulate safety check
        print("  Safety check completed")
        return True

class TaskPlanner:
    """Hierarchical task planner for educational robotics"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_graph = nx.DiGraph()
        self.current_task_id = 0

    def create_task_id(self) -> str:
        """Create a unique task ID"""
        self.current_task_id += 1
        return f"task_{self.current_task_id:03d}"

    def add_task(self, task: Task) -> None:
        """Add a task to the planner"""
        self.tasks[task.id] = task
        self.task_graph.add_node(task.id, task=task)

        # Add dependency edges
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.task_graph.add_edge(dep_id, task.id)

    def create_educational_activity(self, activity_name: str, target_object: str) -> str:
        """Create a complete educational activity with multiple tasks"""
        # Create parent task for the activity
        parent_id = self.create_task_id()
        parent_task = Task(
            id=parent_id,
            name=f"Educational Activity: {activity_name}",
            task_type=TaskType.LEARNING,
            description=f"Complete educational activity: {activity_name}",
            task_status=TaskStatus.PENDING
        )
        self.add_task(parent_task)

        # Create subtasks for the activity
        # 1. Navigate to the object
        nav_id = self.create_task_id()
        nav_task = Task(
            id=nav_id,
            name=f"Navigate to {target_object}",
            task_type=TaskType.NAVIGATION,
            description=f"Navigate the robot to the {target_object}",
            dependencies=[parent_id],
            parameters={'destination': target_object, 'speed': 0.3}
        )
        self.add_task(nav_task)

        # 2. Perceive the object
        per_id = self.create_task_id()
        per_task = Task(
            id=per_id,
            name=f"Perceive {target_object}",
            task_type=TaskType.PERCEPTION,
            description=f"Perceive and analyze the {target_object}",
            dependencies=[nav_id],
            parameters={'target': target_object}
        )
        self.add_task(per_task)

        # 3. Teach about the object
        teach_id = self.create_task_id()
        teach_task = Task(
            id=teach_id,
            name=f"Teach about {target_object}",
            task_type=TaskType.LEARNING,
            description=f"Teach students about the {target_object}",
            dependencies=[per_id],
            parameters={'topic': target_object}
        )
        self.add_task(teach_task)

        # 4. Communicate completion
        comm_id = self.create_task_id()
        comm_task = Task(
            id=comm_id,
            name="Communicate activity completion",
            task_type=TaskType.COMMUNICATION,
            description="Inform students about activity completion",
            dependencies=[teach_id],
            parameters={'message': f"Activity with {target_object} completed!"}
        )
        self.add_task(comm_task)

        return parent_id

    def create_robot_teaching_sequence(self) -> str:
        """Create a complete robot teaching sequence"""
        # Create parent task for the teaching sequence
        parent_id = self.create_task_id()
        parent_task = Task(
            id=parent_id,
            name="Robot Teaching Sequence",
            task_type=TaskType.LEARNING,
            description="Complete robot teaching sequence with multiple objects",
            task_status=TaskStatus.PENDING
        )
        self.add_task(parent_task)

        # Objects to teach about
        teaching_objects = ["book", "ball", "pencil", "robot"]

        prev_task_id = parent_id

        for obj in teaching_objects:
            # Navigate to object
            nav_id = self.create_task_id()
            nav_task = Task(
                id=nav_id,
                name=f"Navigate to {obj}",
                task_type=TaskType.NAVIGATION,
                description=f"Navigate to the {obj}",
                dependencies=[prev_task_id],
                parameters={'destination': obj, 'speed': 0.3}
            )
            self.add_task(nav_task)

            # Perceive object
            per_id = self.create_task_id()
            per_task = Task(
                id=per_id,
                name=f"Perceive {obj}",
                task_type=TaskType.PERCEPTION,
                description=f"Perceive the {obj}",
                dependencies=[nav_id],
                parameters={'target': obj}
            )
            self.add_task(per_task)

            # Teach about object
            teach_id = self.create_task_id()
            teach_task = Task(
                id=teach_id,
                name=f"Teach about {obj}",
                task_type=TaskType.LEARNING,
                description=f"Teach about the {obj}",
                dependencies=[per_id],
                parameters={'topic': obj}
            )
            self.add_task(teach_task)

            # Update previous task for next iteration
            prev_task_id = teach_id

        # Final communication task
        comm_id = self.create_task_id()
        comm_task = Task(
            id=comm_id,
            name="Teaching sequence complete",
            task_type=TaskType.COMMUNICATION,
            description="Communicate completion of teaching sequence",
            dependencies=[prev_task_id],
            parameters={'message': "Teaching sequence completed for all objects!"}
        )
        self.add_task(comm_task)

        return parent_id

    def get_executable_tasks(self) -> List[Task]:
        """Get tasks that are ready to be executed (no pending dependencies)"""
        executable_tasks = []

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                all_deps_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id in self.tasks:
                        dep_task = self.tasks[dep_id]
                        if dep_task.status != TaskStatus.SUCCESS:
                            all_deps_satisfied = False
                            break

                if all_deps_satisfied:
                    executable_tasks.append(task)

        # Sort by priority
        executable_tasks.sort(key=lambda t: t.priority, reverse=True)
        return executable_tasks

    def execute_plan(self, max_concurrent_tasks: int = 1) -> bool:
        """Execute the task plan"""
        print("Starting task plan execution...")

        completed_tasks = 0
        total_tasks = len(self.tasks)

        while completed_tasks < total_tasks:
            # Get executable tasks
            executable_tasks = self.get_executable_tasks()

            if not executable_tasks:
                print("No executable tasks available. Checking for circular dependencies...")
                # Check for potential issues
                if not self._check_plan_status():
                    print("Plan execution blocked by unresolved dependencies")
                    return False
                time.sleep(0.1)
                continue

            # Execute up to max_concurrent_tasks
            tasks_to_execute = executable_tasks[:max_concurrent_tasks]

            for task in tasks_to_execute:
                print(f"\nExecuting: {task.name}")
                task.status = TaskStatus.RUNNING

                success = task.execute()

                if success:
                    task.status = TaskStatus.SUCCESS
                    print(f"✓ Completed: {task.name}")
                    completed_tasks += 1
                else:
                    task.status = TaskStatus.FAILED
                    print(f"✗ Failed: {task.name}")
                    return False

            # Small delay to allow for any async operations
            time.sleep(0.1)

        print(f"\nTask plan completed! {completed_tasks}/{total_tasks} tasks executed successfully.")
        return True

    def _check_plan_status(self) -> bool:
        """Check if the plan is making progress or blocked"""
        pending_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.PENDING)
        running_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING)

        # If all tasks are pending and none are running, check for circular dependencies
        if pending_count > 0 and running_count == 0:
            # Check for circular dependencies in the graph
            try:
                cycles = list(nx.simple_cycles(self.task_graph))
                if cycles:
                    print(f"Circular dependency detected: {cycles}")
                    return False
            except:
                pass  # No cycles found

        return True

    def visualize_plan(self, filename: str = "task_plan.png"):
        """Visualize the task plan as a graph"""
        plt.figure(figsize=(12, 8))

        # Position nodes using a hierarchical layout
        pos = nx.spring_layout(self.task_graph, k=3, iterations=50)

        # Draw the graph
        nx.draw(self.task_graph, pos, with_labels=True, node_color='lightblue',
                node_size=3000, font_size=8, font_weight='bold', arrows=True)

        # Add task details as labels
        labels = {}
        for node_id in self.task_graph.nodes():
            task = self.task_graph.nodes[node_id]['task']
            labels[node_id] = f"{task.name}\n({task.status.value})"

        nx.draw_networkx_labels(self.task_graph, pos, labels, font_size=7)

        plt.title("Educational Robotics Task Plan")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Task plan visualization saved to {filename}")

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get a summary of the current plan"""
        status_counts = {}
        type_counts = {}

        for task in self.tasks.values():
            # Count statuses
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # Count types
            task_type = task.task_type.value
            type_counts[task_type] = type_counts.get(task_type, 0) + 1

        return {
            'total_tasks': len(self.tasks),
            'status_breakdown': status_counts,
            'type_breakdown': type_counts,
            'completed_tasks': status_counts.get('success', 0),
            'pending_tasks': status_counts.get('pending', 0),
            'failed_tasks': status_counts.get('failed', 0)
        }

def main():
    """Main function to demonstrate the task planning system"""
    print("Initializing Educational Robotics Task Planning System")

    # Create task planner
    planner = TaskPlanner()

    # Create an educational activity
    print("\nCreating educational activity...")
    activity_id = planner.create_educational_activity("Object Learning", "red_ball")

    # Create another activity
    print("Creating second educational activity...")
    activity2_id = planner.create_educational_activity("Book Exploration", "book")

    # Create a complete teaching sequence
    print("Creating complete teaching sequence...")
    sequence_id = planner.create_robot_teaching_sequence()

    # Visualize the plan
    planner.visualize_plan("educational_robot_task_plan.png")

    # Print plan summary
    summary = planner.get_plan_summary()
    print(f"\nPlan Summary:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Completed: {summary['completed_tasks']}")
    print(f"  Pending: {summary['pending_tasks']}")
    print(f"  Failed: {summary['failed_tasks']}")

    # Execute the plan
    print("\nExecuting the task plan...")
    success = planner.execute_plan(max_concurrent_tasks=1)

    if success:
        print("\n✓ Task plan executed successfully!")
    else:
        print("\n✗ Task plan execution failed!")

    # Print final summary
    final_summary = planner.get_plan_summary()
    print(f"\nFinal Plan Summary:")
    print(f"  Total tasks: {final_summary['total_tasks']}")
    print(f"  Completed: {final_summary['completed_tasks']}")
    print(f"  Failed: {final_summary['failed_tasks']}")

if __name__ == "__main__":
    main()