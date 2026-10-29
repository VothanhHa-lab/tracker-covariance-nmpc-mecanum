# Experimental protocol: tracker-covariance NMPC for a Mecanum robot among people

Version 1.0, 28/10/2026 (English translation 29/10/2026). The protocol must be **frozen** before the reporting phase
(phase R) is recorded. Every later change is entered in Section 11 (change log) with date and reason.

General principles:
- Every number in the paper comes from rosbag2 files recorded in the phases below, through the scripts
  `tools/analyze_*.py` and `tools/make_macros.py`.
- No number is typed into the paper by hand. No trial is excluded after the results have been seen.
- Data and parameters of the earlier CA-NMPC draft are **not used**, because no original logs exist.

---

## 0. Software preparation

1. **On the computer running the controller** (Ubuntu 22.04, ROS 2 Humble):
   ```bash
   pip install -e pu_nmpc            # core library
   pip install ultralytics            # YOLO for the camera
   ```
   Then copy `ros2_ws/src/*` into a workspace, run `colcon build` and `source install/setup.bash`.
2. **Core tests** (no ROS needed); all must PASS:
   ```bash
   python3 tests/test_core.py
   python3 tests/test_bag.py
   ```
3. **Drivers** (robot bringup, separate): `rplidar_ros` (RPLIDAR A2), `realsense2_camera` with
   `align_depth.enable:=true`, the STM32 bridge that receives `/cmd_vel` and publishes `/odom`, and AMCL or
   `robot_localization` publishing TF `map->odom->base_link` and `/amcl_pose`.
4. **Clock synchronisation** between the robot computer and the ground-truth computer: `chrony`, one machine as server.
   - Record the offset (`chronyc tracking`) at the start and end of each session.
   - **Accept only sessions with |offset| < 2 ms.**
5. **Record versions:**
   - SHA-256 hashes of `pu_nmpc_core/*.py` (function `tools/common.code_hash`) and of `config/*.yaml`, entered in the
     lab notebook for every session;
   - CPU/RAM model of the computer running the NMPC.

## 1. P1: Test site and ground-truth validation

1. **Map and test area.** Build a map (SLAM Toolbox) and fix the `map` frame. Mark the test area on the floor
   (corridor and hall), with dimensions measured by laser meter.
2. **Ground-truth camera.** Mount a fixed ceiling camera covering the whole test area. Calibrate its intrinsics with a
   checkerboard (store the `camera_info` file).
3. **Floor tags.** Place >= 5 AprilTag 36h11 tags on the floor. Measure their coordinates in `map` and enter them in
   `config/ground_truth.yaml`.
4. **Ground-truth validation.** Place a target tag at >= 9 measured points at two heights (0.8 m and 1.45 m) and record
   10 s per point (sessions `G01`...).
   - Declare these points in `config/gt_validation_points.yaml`.
   - Run `tools/analyze_gt_validation.py` -> `results/gt_validation.json`.
5. **Acceptance criterion:** ground-truth RMSE <= 3 cm. Otherwise improve the ground-truth system; do not proceed to
   the next phases.

## 2. P2: Sensor noise, detection rate, delay and pedestrian motion

The robot is **stationary** with motors off. Each participant wears a tag on a cap; the tag height is measured for
each person.

- **Sessions `N01`...`N06`** (about 3 min each, `sensor_noise.launch.py`):
  - people walk along lines drawn on the floor covering ranges of 0.5-7 m and angles of +/-180 deg, at slow, normal and
    fast speed;
  - one session with two people crossing and occluding each other.
- **Standing still:** at least 2 sessions with people standing at marks of 1, 2, ..., 7 m, to separate noise from
  motion-induced error.
- **Analysis:**
  ```bash
  python3 tools/analyze_sensor_noise.py --gt-std <ground-truth RMSE from P1> data/N0*/csv
  ```
  Outputs:
  - noise model sigma_r = a + b*r, sigma_b;
  - detection rate versus range;
  - false detections per scan;
  - effective delay;
  - the CV-model parameter q for walking people.
- **Configuration update:** check `config/sensors_identified.yaml`, then copy it to `config/sensors.yaml`. Set
  `q_acc = q_acc_pred =` the value `pedestrian_q`. Set `lag` >= the largest observed delay.

## 3. P3: Identification of the velocity loops

**Mechanical measurements.** Measure the wheel radius, lx, ly and the enclosing radius of the robot. Enter them in
`config/robot.yaml` and change `source` to `measured`.

**Step experiments** (`identification.launch.py`, sessions `I01`-`I03`, on an empty floor with an operator at the e-stop):
- Enable the safety gate: `ros2 topic pub /trial/enable std_msgs/Bool "data: true"`.
- Step sequences in vx, vy and omega at several amplitudes.

**Analysis:** `tools/analyze_identification.py data/I0*/csv` -> tau, K and delay per axis, with the NRMSE on the last
third of each session.
- **Acceptance:** NRMSE <= 10 %. Otherwise report it as an adverse result and consider a higher-order model.
- Update `tau` in `config/robot.yaml`.

**Wheels and accelerations.** Determine `wheel_speed_max` (80 % of the measured saturation) and `body_acc_max` (no wheel
slip, checked against ground truth).

## 4. P4: Tracker evaluation

Run offline on all sessions with ground truth (P2 and the closed-loop trials):
```bash
python3 tools/analyze_tracking.py --q-pred <pedestrian_q> data/N0*/csv data/R*/csv
```

Compared configurations:
- fusion, LiDAR only, camera only;
- each with and without delay compensation.

Metrics:
- position RMSE, OSPA (c = 1 m), detection rate, ID switches;
- **coverage of the 95 % ellipse at horizons of 0, 0.5 and 1.0 s**.

**Key criterion:** the 1-s coverage of the fusion configuration with delay compensation must be >= 90 %. Otherwise the
covariance is optimistic; q may then be adjusted **on P2 data only** (never on phase-R data) and frozen again. Both
attempts must be reported.

## 5. P5: Pilot (phase D)

- Pilot schedule:
  ```bash
  python3 tools/make_schedule.py --phase D --participants 2 --reps 1 --seed <seed D>
  ```
- Purpose: check safety, computation time on the real computer, software faults and scenario difficulty.
- **Allowed** adjustments: v_ref, timeouts, starting positions of people, and NMPC weights **common to all methods**.
- **Not allowed:** adjustments specific to the proposed method.
- Phase-D data are **not reported** as main results; the methods section may state "a pilot of n trials was used to
  adjust ...".
- **Parameter freeze.** After the pilot, freeze `config/*.yaml` and enter the hashes in Section 11. No further changes.

## 6. P6: Reporting campaign (phase R)

### 6.1 Design
- **Methods:** cov (proposed), fixed and heuristic (reproduction of CA-NMPC). All share the NMPC, the cost and the limits.
- **Scenarios:**
  - H1: crossing
  - H2: head-on in a corridor
  - H3: narrow passage with two people
  - H4: person stepping out from behind an obstacle
  - H5: approach outside the camera field of view
- **Sample size:** 5 people x 4 repetitions = **20 trials per cell**, 300 trials in total.
  - With 0 contacts in 20 trials the 95 % upper bound is still 13.9 %, so the paper must **not** claim a low contact
    probability. The main aim is to measure clearance, personal-zone time, success rate and time.
  - If more participants are available, increase the number of people, not the repetitions per person.
- **Randomisation:**
  ```bash
  python3 tools/make_schedule.py --phase R --participants 5 --reps 4 --seed <seed R>
  ```
  The scenario order is shuffled per person; the method order rotates as a Latin square.
- **Blinding:** participants do not know which method is running. The operator knows and must not instruct
  participants differently between methods.

### 6.2 Procedure of one trial
1. Check the ground-truth lighting, chrony and battery. Fill in the trial sheet (Section 9).
2. Start:
   ```bash
   ros2 launch pu_nmpc_ros closed_loop.launch.py trial_id:=R017 method:=<per schedule> schedule:=protocol/schedule_R.csv
   ```
   The ground-truth node runs on the ceiling-camera computer with the same ROS_DOMAIN_ID.
3. The participant stands at the starting position of the scenario. The operator opens `mark_event` and presses `s`.
   The participant starts walking at the light signal or line.
4. The trial ends when the robot reaches the goal, at the timeout, or when the operator presses `i` (intervention:
   e-stop or manual stop) or `a` (abort for a technical fault).
5. Every collision or touch is logged with `n` -> `note:contact` and on the sheet.

### 6.3 Failed trials
- Trials aborted for a technical fault not caused by the controller (ground-truth loss, bag drop, battery) are **re-run
  once** at the end of the session with the same trial_id plus suffix `b`, and the reason is recorded.
- The original trial is kept and reported in the table of aborted trials.
- Trials with an intervention are **not** re-run and count as failures.

## 7. Analysis plan (fixed in advance)

- **Step 1:** extraction: `tools/bag_to_csv.py data/<id>/bag` for every session.
- **Step 2:** metrics:
  ```bash
  python3 tools/analyze_closed_loop.py --phase R data
  ```
  This lists **every** scheduled trial, including trials that were not run.
- **Step 3:** macros for the paper:
  ```bash
  python3 tools/make_macros.py --paper ../paper_sensors
  ```

Metric definitions:
- **Primary outcomes:** minimum clearance (robot edge to person edge, from ground truth) and time inside the 0.5-m
  personal zone.
- **Secondary outcomes:** success rate, time to goal, jerk, computation time (p50, p95, max).
- **Contact:** centre distance below r_robot + 0.25 m, or an operator note `note:contact`.

Statistics:
- **Paired comparisons** by (person, scenario, repetition): cov - fixed and cov - heuristic, with 95 % bootstrap
  intervals (10 000 resamples).
- **Proportions:** two-sided Clopper-Pearson intervals; with 0 events, the one-sided upper bound.
- **No multiple-hypothesis testing** for individual scenarios; per-scenario results are descriptive.

Adverse results:
- Report them in full, including cases where cov loses to fixed in time or success rate; by design the method is more
  cautious when uncertainty is large.
- Report stops caused by `stale_stop`, `watchdog_brake` or `solver_fail_brake`.

## 8. Safety and ethics

- A decision of the institutional ethics or scientific committee is required **before P2** (P2 already involves
  participants), or a written exemption with reasons.
- Participants are adults, take part voluntarily, sign the consent form (Section 10) and may withdraw at any time.
- Data are pseudonymised with codes P01...; no names are stored in the bags.
  - Colour images are used only by the detector and are **not recorded**: the image topic is not in the recording list.
  - If images are needed for illustration, separate consent is required and faces are blurred.
- **Safety limits:**
  - v_ref <= 0.6 m/s;
  - the operator holds a wireless e-stop throughout each trial;
  - every trial has a 0.3-m hard-stop zone on the raw scan at the motor bridge, independent of the NMPC (checked before
    each session);
  - participants wear closed shoes and are instructed not to run.
- **Before each session:** test the e-stop three times, and test the watchdog by stopping `/odom`: the robot must stop.

## 9. Trial sheet (print or spreadsheet)

Trial ID | Date/time | Person (Pxx) | Scenario | Method | Repetition | chrony offset (ms) | Battery (%) | Outcome (goal / timeout / intervention / abort) | Contact (yes/no) | Notes | Operator

## 10. Consent form (summary of required content)

- Project title, institution, responsible person and contact phone number.
- Purpose, what participants do (walk along prescribed lines near the robot, about 1-2 h per session), risks (light
  contact at <= 0.6 m/s) and the safety measures.
- Data collected: trajectories, LiDAR scans, depth images; colour images are not stored. Pseudonymisation, retention
  period and storage location.
- Right to withdraw at any time without giving a reason.
- Signatures and dates of the participant and the researcher.

## 11. Change log and parameter freeze

| Date | Change | Code / config hash | Recorded by |
|---|---|---|---|
| 28/10/2026 | Version 1.0 | - | - |
| 29/10/2026 | English translation (no change of content) | - | - |
