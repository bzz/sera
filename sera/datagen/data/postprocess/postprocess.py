
import argparse
import json
import os
import yaml

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict

from sera.config_schema import PostprocessConfig
from sera.datagen.data.postprocess.utils import (
    add_train_key,
    reformat_assistant_message,
    transform_traj_xml,
    transform_traj_hermes,
    transform_traj_raw
)
from sera.constants import (
    HERMES_DEFAULT_SYSTEM_PROMPT,
    XML_DEFAULT_SYSTEM_PROMPT,
    SWEAGENT_TOOLS,
    MINI_SWEAGENT_HERMES_SYSTEM_PROMPT,
    MINI_SWEAGENT_XML_SYSTEM_PROMPT,
    MINI_SWEAGENT_TOOLS,
)
from sera.utils import dump_jsonl, ExperimentFolder

MAP_TO_PARSER = {
    "xml": transform_traj_xml,
    "hermes": transform_traj_hermes,
    "raw": transform_traj_raw
}

MAP_TO_SYSTEM_PROMPT = {
    "sweagent": {
        "xml": XML_DEFAULT_SYSTEM_PROMPT,
        "hermes": HERMES_DEFAULT_SYSTEM_PROMPT,
        "raw": None,
    },
    "mini-swe-agent": {
        "xml": MINI_SWEAGENT_XML_SYSTEM_PROMPT,
        "hermes": MINI_SWEAGENT_HERMES_SYSTEM_PROMPT,
        "raw": None,
    },
}

MAP_TO_TOOL_JSON = {
    "sweagent": SWEAGENT_TOOLS,
    "mini-swe-agent": MINI_SWEAGENT_TOOLS,
}

def _normalize_mini_sweagent_traj(traj):
    """Normalize mini-swe-agent traj format to match SWE-agent format for transform functions."""
    normalized = dict(traj)
    normalized["history"] = []
    for msg in traj["messages"]:
        if msg["role"] == "exit":
            continue
        msg = dict(msg)
        if msg["role"] == "assistant" and msg.get("content") is None:
            msg["content"] = ""
        normalized["history"].append(msg)
    return normalized

def get_raw_trajectories(traj_dir: Path, report: Dict, tool_call_format: str, add_think: bool, enforce_submit: bool, tool_json: bool, agent_harness: str = "sweagent"):
    transform_traj = MAP_TO_PARSER[tool_call_format]
    system_prompt = MAP_TO_SYSTEM_PROMPT[agent_harness][tool_call_format]
    tools_json = MAP_TO_TOOL_JSON[agent_harness]

    def _process_folder(folder):
        if report and folder not in report["resolved_ids"]:
            return None

        synth_path = os.path.join(traj_dir, folder, f"{folder}.synth")
        if os.path.exists(synth_path): # If this is from stage one, we confirm that the patch addressed the initial prompt
            try:
                with open(synth_path, "r") as synth_f:
                    synth_json = json.load(synth_f)
            except json.JSONDecodeError:
                return None
            if not synth_json["is_good_patch"]:
                return None

        # Handle different traj file extensions
        if agent_harness == "mini-swe-agent":
            traj_path = os.path.join(traj_dir, folder, f"{folder}.traj.json")
        else:
            traj_path = os.path.join(traj_dir, folder, f"{folder}.traj")
        if not os.path.exists(traj_path):
            return None

        try:
            raw_traj_json = json.load(open(traj_path, "r"))
        except json.JSONDecodeError:
            return None

        # Handle different exit status casing
        exit_status = raw_traj_json["info"].get("exit_status", "")
        if enforce_submit and exit_status.lower() != "submitted":
            return None

        # Normalize mini-swe-agent format to match what transforms expect
        if agent_harness == "mini-swe-agent":
            raw_traj_json = _normalize_mini_sweagent_traj(raw_traj_json)

        traj = transform_traj(raw_traj_json, system_prompt, add_think=add_think)
        if not traj:
            print(f"Error transforming {folder}")
            return None

        traj["instance_id"] = folder
        if tool_json:
            traj["tool_json"] = tools_json
        return traj

    # Convert folders to data trajectories
    folders = [
        x for x in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, x))
    ]
    print(f"Found {len(folders)} trajectory folders in {traj_dir}")
    trajs = []
    with ThreadPoolExecutor() as ex:
        futures = [ex.submit(_process_folder, folder) for folder in folders]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            traj = fut.result()
            if traj is None:
                continue
            trajs.append(traj)

    return trajs

def create_file_name(config: PostprocessConfig, traj_dir: Path, report_path: Path):
    report_path_name = report_path.stem if report_path else "noreport"
    return "_".join([traj_dir.name,
                    report_path_name,
                    f"addthink-{config.add_think}",
                    f"atk-{config.add_train_key}",
                    f"rft-{bool(config.reformat_assistant_message) and config.tool_call_format == 'hermes'}",
                    f"format-{config.tool_call_format}"]) + ".jsonl"

def format_and_save(
    config: PostprocessConfig,
    traj_dir: Path,
    report_path: Optional[Path],
    out_dir: Path,
    agent_harness: str = "sweagent",
):
    report = None
    if report_path:
        print("Only keeping trajectories for resolved instances")
        with open(report_path, "r") as f:
            report = json.load(f)
    dataset = get_raw_trajectories(traj_dir=traj_dir,
                                            report=report,
                                            tool_call_format=config.tool_call_format,
                                            add_think=config.add_think,
                                            enforce_submit=config.enforce_submit,
                                            tool_json=config.include_tool_json,
                                            agent_harness=agent_harness)
    print(f"Found {len(dataset)} valid trajectories")

    if config.add_train_key:
        dataset = add_train_key(dataset=dataset)
    if config.reformat_assistant_message and config.tool_call_format == "hermes": # Only works for hermes right now :(
        dataset = reformat_assistant_message(dataset=dataset, mode=config.reformat_assistant_message)

    save_fp = out_dir / create_file_name(config=config, traj_dir=traj_dir, report_path=report_path)
    dump_jsonl(fp=save_fp, data=dataset, overwrite=True)
    return save_fp
