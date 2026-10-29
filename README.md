# pu_nmpc: tracker-covariance safety margins for NMPC of a Mecanum robot among people

Code, configuration, per-run results and analysis scripts for the manuscript

> Vo Thanh Ha, Vo Quang Vinh, Nguyen Duy Trung. *Tracker-Covariance Safety Margins with Delay Compensation for
> Nonlinear Model Predictive Control of a Mecanum Robot Among People: A Simulation Study.* Submitted to *Sensors* (MDPI).

The repository also contains ROS 2 nodes and an experimental protocol for validating the method on the real platform.
All results in the manuscript come from simulation (and a partial check on public real LiDAR data, see below); no
hardware experiment has been carried out yet.

**Reproducibility.** The tag [`v1.0-paper-results`](../../tree/v1.0-paper-results) marks the exact code, configuration
and result files used for the manuscript. Later commits translate comments, docstrings and console messages into
English without changing program logic (checked by comparing the syntax trees of all Python files and the parsed YAML
values). The campaign metadata files (`results/*.meta.json`) store hashes of the code and configuration at the time of
each run; these hashes refer to the tagged version. Log files in `results/*_log.txt` are the raw console output of the
original runs and therefore contain Vietnamese progress messages.

## Layout

| Folder | Content |
|---|---|
| `pu_nmpc_core/` | Pure-Python library, testable without ROS: robot model (`models`), asynchronous people tracker (`tracker`), people detection (`detect`), covariance constraint (`constraint`), CasADi/IPOPT NMPC (`nmpc`), controller with all compared methods (`controller`), identification (`identification`), metrics and statistics (`metrics`), simulation-study environment (`simstudy`), software-test simulator (`sim`) |
| `config/` | YAML parameters. `simstudy.yaml` lists every simulation parameter with its source (T5 = platform specification, DS = data sheet, AS = assumed). Entries marked `TO_MEASURE` / `TO_IDENTIFY` in the hardware configs must be replaced by measured values before experiments |
| `tools/` | Simulation campaigns (`sim_campaign.py`), analysis (`analyze_sim.py`, `analyze_drow.py`, `calibrate_inflation.py`), LaTeX macros and figures for the paper (`make_sim_macros.py`, `make_sim_figures.py`, `make_sim_traces.py`), and the hardware pipeline (`make_schedule`, `bag_to_csv`, `analyze_*`, `make_macros`, `make_figures`) |
| `results/` | Per-run CSV files of every campaign, their metadata (`*.meta.json`), `sim_summary.json`, `calibration.json`, `drow_validation.json`, the design log `sim_design_log.md` and raw console logs |
| `ros2_ws/src/pu_nmpc_msgs` | Messages: detections, tracked people, controller diagnostics |
| `ros2_ws/src/pu_nmpc_ros` | Nodes: `lidar_people_detector`, `camera_people_detector`, `fusion_tracker`, `nmpc_controller`, `scan_stop_guard`, `gt_tag_tracker`, `trial_manager`, `step_test`, `mark_event`; launch files for each protocol phase |
| `protocol/` | Experimental protocol P0-P7 for the hardware validation (safety, ethics, trial sheet, parameter-freeze log) |
| `tests/` | `test_core.py` (core tests), `test_bag.py` (bag write/read), `make_synthetic_sessions.py` (**synthetic** data for pipeline tests only) |

## Installation

```bash
pip install -e .[analysis]      # numpy, scipy, casadi, pyyaml, pandas, matplotlib, rosbags
python3 tests/test_core.py
```

## Reproducing the simulation study

```bash
./run_sim_campaigns.sh          # main (1500 runs) + robustness (2250) + ablation (1500); ~4 h with 7 processes
python3 tools/sim_campaign.py --campaign devcal && python3 tools/calibrate_inflation.py   # gamma on development seeds
./run_revision.sh               # post hoc campaign (new seeds 3000-3049) + single-process timing
./run_alt.sh                    # CC-const baseline (main and new seeds) + its timing
./run_timing2.sh                # repeated single-process timing
python3 tools/analyze_sim.py
python3 tools/make_sim_macros.py --paper ../paper_sensors
python3 tools/make_sim_figures.py --paper ../paper_sensors
python3 tools/make_sim_traces.py --paper ../paper_sensors      # example runs on seed 1000
```

Seeds 0-49 are development seeds; all reported runs use seeds that were not used during development
(1000-1099, 2000-2049, 3000-3049). The decisions taken on development seeds and the post hoc additions are listed,
in order, in `results/sim_design_log.md`.

Note on exact reproducibility: the NMPC solver stops at a CPU-time limit (80 ms), so a single run can differ slightly
between machines or under different load; campaign statistics are not affected in practice. `make_sim_traces.py`
reports any difference between a re-run and the stored result.

## Partial check on public real data (DROW v2)

The DROW v2 dataset (Beyer et al., *IEEE RA-L* 2018, MIT licence; the `reha` subset is CC-BY-NC-SA 3.0) is not
included (about 1.1 GB unpacked). To reproduce the check:

```bash
mkdir -p external_data/DROWv2 && cd external_data/DROWv2
curl -L -o DROWv2-data.zip https://github.com/VisualComputingInstitute/DROW/releases/download/v2/DROWv2-data.zip
unzip DROWv2-data.zip && cd ../..
python3 -I tools/analyze_drow.py external_data/DROWv2/DROWv2-data     # -> results/drow_validation.json
```

SHA-256 of the zip file used: `1537422ab02e14de0cbd60dfde60ca9efb354e9d48e4d7bcca5ecfcdbe0df227`.
The sensor (SICK S300, 225 deg, 12.5 Hz) differs from the RPLIDAR A2 of the target robot, the labels are hand-set
positions rather than an independent ground truth, and no static map is available, so the false-detection count is an
upper bound.

## Data-integrity safeguards built into the code

- **Provenance.** Every result file records hashes of its inputs and of the source code, and a `synthetic` flag.
- **Synthetic data are blocked.** The macro and figure scripts refuse to write into the paper folder if any result is
  synthetic; missing results are printed as `\TBD{}`.
- **Design parameters go through macros too.** Values such as d0, eps, N, dt and v_ref are read from `config/`, so
  paper and code always agree.
- **All trials are listed.** The hardware analysis lists every scheduled trial, including aborted or missing ones.

## Status

- Verified on the development machine: core tests, bag read/write with the custom messages, the full analysis pipeline
  on synthetic sessions (recovering the injected time constants, delays and noise), and all simulation campaigns.
- Not verified: the ROS 2 nodes are only syntax-checked (no ROS 2 on the development machine); they must be built with
  colcon and tested on the robot in protocol phases P0 and P5. Solve times were measured on an Apple M3 only.

## Licence

MIT, see `LICENSE`.
