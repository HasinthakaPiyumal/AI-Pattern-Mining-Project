import random

def simulate_expert_trajectory(problem_id):
    """Generates a single simulated expert trajectory for cloud infrastructure troubleshooting.

    A trajectory consists of a sequence of problem, rationale, tool call, and tool output.
    """
    problems = [
        ("VM instance unresponsive", "instance-123", {"vm_id": "instance-123", "log_type": "syslog"}, {"status": "ERROR", "message": "Disk full on /var"}, "resize_disk"),
        ("Database connection failing", "db-server-456", {"service_name": "database"}, {"status": "DOWN", "message": "Service not running"}, "restart_service"),
        ("High latency in API gateway", "api-gateway-789", {"vm_id": "api-gateway-789", "log_type": "network"}, {"status": "OK", "message": "No network errors, checking load"}, "scale_up_instances"),
    ]

    problem_description, entity_id, initial_tool_args, initial_tool_output, next_action = random.choice(problems)

    trajectory = []
    # Initial problem and expert's first thought/action
    trajectory.append({
        "problem": problem_description,
        "entity_id": entity_id,
        "rationale": f"User reported '{problem_description}'. First, I will check logs for {entity_id}.",
        "tool_call": {"tool_name": f"check_{entity_id.split('-')[0]}_logs", "args": initial_tool_args},
        "tool_output": initial_tool_output # Output from the first tool call
    })

    # Subsequent steps based on the initial output
    if "Disk full" in initial_tool_output.get("message", ""):
        trajectory.append({
            "problem": problem_description,
            "entity_id": entity_id,
            "rationale": f"Logs indicate disk full on {entity_id}. Resizing disk to resolve.",
            "tool_call": {"tool_name": "resize_disk", "args": {"vm_id": entity_id, "size": "200GB"}},
            "tool_output": {"status": "SUCCESS", "message": "Disk resized."}
        })
        trajectory.append({
            "problem": problem_description,
            "entity_id": entity_id,
            "rationale": f"Disk resized on {entity_id}. Problem should be resolved.",
            "tool_call": None,
            "tool_output": {"status": "RESOLVED", "message": "Issue resolved."}
        })
    elif "Service not running" in initial_tool_output.get("message", ""):
        trajectory.append({
            "problem": problem_description,
            "entity_id": entity_id,
            "rationale": f"Service is down on {entity_id}. Attempting to restart the service.",
            "tool_call": {"tool_name": "restart_service", "args": {"service_name": entity_id.split('-')[0]}},
            "tool_output": {"status": "SUCCESS", "message": "Service restarted."}
        })
        trajectory.append({
            "problem": problem_description,
            "entity_id": entity_id,
            "rationale": f"Service restarted on {entity_id}. Problem should be resolved.",
            "tool_call": None,
            "tool_output": {"status": "RESOLVED", "message": "Issue resolved."}
        })
    elif "High latency" in problem_description and "load" in initial_tool_output.get("message", ""):
         trajectory.append({
            "problem": problem_description,
            "entity_id": entity_id,
            "rationale": f"Network logs clean, but high latency persists. Scaling up instances for {entity_id}.",
            "tool_call": {"tool_name": "scale_up_instances", "args": {"service_name": entity_id.split('-')[0], "count": 2}},
            "tool_output": {"status": "SUCCESS", "message": "Instances scaled up."}
        })
         trajectory.append({
            "problem": problem_description,
            "entity_id": entity_id,
            "rationale": f"Instances scaled up for {entity_id}. Problem should be resolved.",
            "tool_call": None,
            "tool_output": {"status": "RESOLVED", "message": "Issue resolved."}
        })
    else:
        trajectory.append({
            "problem": problem_description,
            "entity_id": entity_id,
            "rationale": f"Further investigation needed for {entity_id}. Escalating.",
            "tool_call": None,
            "tool_output": {"status": "ESCALATED", "message": "Escalated to human expert."}
        })

    return trajectory

def generate_dataset(num_trajectories):
    """Generates a dataset of simulated expert trajectories."""
    dataset = []
    for i in range(num_trajectories):
        dataset.append(simulate_expert_trajectory(i))
    return dataset

if __name__ == "__main__":
    # Example usage
    sample_dataset = generate_dataset(3)
    for i, trajectory in enumerate(sample_dataset):
        print(f"\n--- Trajectory {i+1} ---")
        for step in trajectory:
            print(step)