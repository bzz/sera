import json
import os

from pathlib import Path
from tqdm import tqdm

from sera.config_schema import EvalConfig
from sera.utils import (
    get_sweagent_patch,
    get_mini_sweagent_patch,
    load_yaml
)

def compare_patch_recall(target_patch, produced_patch, threshold):
    target_patch_changes = []
    for line in target_patch.splitlines():
        if line.startswith(("diff", "---", "+++", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            if line[1:].strip() == "": # skip blank lines
                continue
            target_patch_changes.append(line)

    # Find all changed lines
    produced_patch_changes = []
    for line in produced_patch.splitlines():
        if line.startswith(("diff", "---", "+++", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            if line[1:].strip() == "": # Skip empty lines, spaces, etc.
                continue
            produced_patch_changes.append(line)

    # Calculate recall
    match_count = 0
    for line in target_patch_changes:
        if line in produced_patch_changes:
            match_count += 1
            produced_patch_changes.remove(line)
    if len(target_patch_changes) == 0:
        return None
    return (match_count / len(target_patch_changes)) >= threshold

def eval_loop(config: EvalConfig, instances_fp: Path, second_stage_dir: Path, agent_harness: str):
    resolved_instances = []
    pred_json_found = os.path.exists(os.path.join(second_stage_dir, "preds.json"))
    if pred_json_found:
        with open(os.path.join(second_stage_dir, "preds.json"), "r") as f:
            preds = json.load(f)
    instances = load_yaml(instances_fp)
    for instance in tqdm(instances):
        instance_id = instance["id"]
        traj_path = os.path.join(second_stage_dir, instance_id, f"{instance_id}.traj")
        if pred_json_found:
            if instance_id not in preds:
                print("no pred json found")
                continue
            model_patch = preds[instance_id]["model_patch"]
        else:
            if agent_harness == "sweagent":
                model_patch = get_sweagent_patch(second_stage_dir, instance_id)
            elif agent_harness == "mini-swe-agent":
                model_patch = get_mini_sweagent_patch(second_stage_dir, instance_id)
            else:
                raise RuntimeError("Agent harness must be sweagent | mini-swe-agent")
        if not model_patch:
            print("No edit was made in the second stage")
            continue
        if not instance["extra_fields"].get("pred_patch", ""):
            print(f"No target patch in instance file")
            continue
        resolved = compare_patch_recall(target_patch=instance["extra_fields"]["pred_patch"],
                                        produced_patch=model_patch,
                                        threshold=config.compare_patch_threshold)
        if resolved:
            resolved_instances.append(instance_id)
    return resolved_instances
