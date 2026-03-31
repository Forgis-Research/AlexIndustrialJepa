# Overnight Data Curation Prompt

Copy this entire prompt when starting the overnight session on the VM.

---

## Launch Command

```bash
# Create new tmux session
tmux new -s data-curation

# Navigate to project
cd ~/dev/IndustrialJEPA/mechanical-datasets

# Start Claude with the prompt below
claude
```

Then paste the following prompt:

---

## PROMPT (copy everything below this line)

```
You are running an overnight data curation session. Your goal is to build a comprehensive mechanical vibration dataset on HuggingFace.

## CRITICAL INSTRUCTIONS

1. **ALWAYS use the data-curator agent** for all data tasks:
   - Use Task tool with subagent_type="data-curator" for downloads, processing, validation, uploads
   - The data-curator agent has the right tools and expertise

2. **DISK CONSTRAINT: MAX 10GB at any time**
   - Process one dataset at a time
   - Upload to HuggingFace, then DELETE local files before next dataset
   - Monitor with: du -sh raw/ processed/

3. **Target**: https://huggingface.co/datasets/Forgis/Mechanical-Components

## TWO-LEVEL SCHEMA (IMPORTANT!)

We use a two-level schema to avoid duplicating source-level metadata:

**Level 1: source_metadata config** (one row per source dataset)
- source_id, full_name, url, license
- sampling_rate_hz, signal_duration_sec (constant per source)
- component_type, manufacturer, model
- has_episodes, has_transitions, has_continuous_severity flags

**Level 2: bearings/gearboxes/motors configs** (per-sample data)
- source_id (foreign key to source_metadata)
- signal, n_channels, channel_names
- health_state, fault_type, fault_severity
- rpm, load, load_unit
- episode_id, episode_position, rul_percent (for prognostics)
- is_transition, transition_type (for action-conditioning)

**First task**: Create the source_metadata config with entries for all 10 datasets.

## YOUR TASK

Read these files first:
- mechanical-datasets/OVERNIGHT_TASK.md (full plan and schema)
- mechanical-datasets/datasets_inventory.md (10 datasets with links)

Then:

1. **FIRST**: Create source_metadata config with all 10 source datasets
2. Process datasets in order (smallest first):

| # | Dataset | Size | Config | Has Episodes | Has Transitions |
|---|---------|------|--------|--------------|-----------------|
| 1 | MFPT | ~100MB | bearings | No | No |
| 2 | OEDI | ~500MB | gearboxes | No | No |
| 3 | PHM 2009 | ~1GB | gearboxes | No | No |
| 4 | FEMTO | ~2GB | bearings | Yes | No |
| 5 | MCC5-THU | ~2GB | gearboxes | Yes | **Yes** |
| 6 | Mendeley | ~3GB | bearings | Yes | **Yes** |
| 7 | XJTU-SY | ~5GB | bearings | Yes | No |
| 8 | IMS | ~6GB | bearings | Yes | No |
| 9 | Paderborn | ~20GB | bearings | No | No |

**Note**: CWRU already uploaded. MCC5-THU and Mendeley have transitions - prioritize these for action-conditioning!

## FOR EACH DATASET

Use the data-curator agent to:
1. Download to raw/{dataset_name}/
2. Explore data structure, understand format
3. Convert to two-level schema:
   - Update source_metadata if not already done
   - Create samples with source_id foreign key
4. For prognostics datasets (IMS, XJTU-SY, FEMTO): populate episode_id, episode_position, rul_percent
5. For transition datasets (MCC5-THU, Mendeley): populate is_transition, transition_type
6. Validate: check for NaN, signal stats, label distribution
7. Upload to HuggingFace (append to existing config)
8. DELETE local raw/ and processed/ files
9. Log progress to progress.log
10. Move to next dataset

## COMPONENTS & SENSORS

**Components (include):**
- Bearings (ball, roller)
- Gears (spur, helical, bevel)
- Gearboxes (assemblies)
- Motor bearings (context: electric_motor)

**Components (exclude for now):**
- Pumps, valves, hydraulics

**Sensors (include ALL available):**
- Accelerometers / vibration
- Current sensors (motor current)
- Tachometer / RPM
- Temperature sensors
- Acoustic emission

**Multi-modal handling:**
- Store all sensor types in same sample
- Use `channel_names` to describe each: ["accel_x", "accel_y", "current_A", "temp"]
- Use `channel_modalities` to tag type: ["vibration", "vibration", "current", "temperature"]
- For slow signals (temperature): use `slow_signals` dict if rate differs significantly

## ENVIRONMENT

The .env file is at: ~/dev/IndustrialJEPA/.env
Load with: from dotenv import load_dotenv; load_dotenv("path/to/.env")

Contains: HF_TOKEN, HF_DATASET_REPO

## EXISTING SCRIPTS

Check mechanical-datasets/scripts/ for reusable code - adapt for other datasets.

## SUCCESS CRITERIA

By morning:
- source_metadata config created with all sources
- At least 5 datasets uploaded with proper schema
- Episode/transition fields populated where available
- Progress logged
- Disk usage never exceeded 10GB

## IF SOMETHING FAILS

- Log the error to progress.log
- Skip that dataset, move to next
- Try alternative download links (Kaggle mirrors often work)
- Don't get stuck - keep making progress

Work autonomously through the night. Good luck!
```

---

## Quick Reference

| Dataset | Size | Config | Download Link |
|---------|------|--------|---------------|
| MFPT | 100MB | bearings | mfpt.org or GitHub mirror |
| CWRU | 500MB | bearings | engineering.case.edu (DONE) |
| OEDI | 500MB | gearboxes | data.openei.org |
| PHM 2009 | 1GB | gearboxes | phmsociety.org |
| FEMTO | 2GB | bearings | PHM 2012 / Kaggle |
| MCC5-THU | 2GB | gearboxes | GitHub |
| Mendeley | 3GB | bearings | data.mendeley.com |
| XJTU-SY | 5GB | bearings | GitHub/MediaFire |
| IMS | 6GB | bearings | data.nasa.gov |
| Paderborn | 20GB | bearings | uni-paderborn.de |
