# Design log of the simulation study (frozen 2026-10-29 07:15)

All decisions below were taken on development seeds 0-9 (`sim_campaign --campaign dev`) BEFORE the reporting seeds
were run (main 1000-1099, robustness 2000-2049, ablation 1000-1099). No parameter was changed on the basis of
reporting results.

1. Corridor H2 widened from 2.0 m to 2.6 m; H3 uses a 3.0-m corridor with a 1.6-m opening. Reason: with non-reactive
   people a 2.0-m corridor leaves no room for both parties to pass (contacts independent of the controller).
2. Violations of wall constraints are penalised 100 times more than person constraints (all three methods).
3. Pedestrian model q = 0.022 m^2/s^3, maximum-likelihood estimate on 300 pedestrian trajectories of seeds 0-49
   (replacing the hand-set value 0.5).
4. Contacts are classified as active/passive; wall contact is checked against the 0.60 x 0.55 m rectangular footprint.
5. Nominal pedestrians avoid the robot with a social force (A = 3 m/s^2, B = 0.5 m, lambda = 0.35, assumed);
   non-reactive pedestrians are kept as a worst-case condition. Reason: with non-reactive pedestrians all pilot contacts
   were passive (robot already stopped) and did not reflect the controller.
6. Static points: the nearest point in each angular sector, 16 sectors (8 sectors missed the concave corners of the H3
   opening).
7. Actuation-delay compensation in the NMPC (input_delay = assumed actuation delay 0.05 s), for all three methods.

Final pilot (seeds 0-9, 150 runs): 150/150 successes for all three methods. Pilot results are NOT reported.

Hashes: code=ca4e993fe8c95636 simstudy.yaml=44783b41c653ecbd controller.yaml=5827ce9c3969c872 tracker.yaml=12753bf8fef76ff8

## Post hoc additions (29/10/2026)

These additions were decided after the reporting results had been seen, and the paper labels them as post hoc. They
are evaluated on new seeds 3000-3049 that had never been run.

1. **Covariance inflation gamma.** Rule fixed before the revision campaign: choose the smallest gamma in the grid
   (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4) for which the pooled coverage reaches >= 95 % at all three horizons 0, 0.5 and 1 s.
   The choice uses development seeds 0-9 only (campaign devcal, method cov). Result: gamma = 1.5
   (`results/calibration.json`).
2. **Fixed-margin sweep.** Buffers of 0.45, 0.60 and 0.75 m in addition to the original 0.30 m, for a comparison on the
   clearance-time trade-off.
3. **Solve-time measurement.** Timing campaign with one process and no other campaign running (seeds 3000-3009).
4. **Code changes.** `MethodParams.cov_inflate` (default 1, so earlier campaigns are unchanged) and coverage counters for
   the gamma grid in `simstudy.py`.
5. **Fourth baseline "ccconst"** (requested by the corresponding author on 29/10/2026). The same chance constraint as
   the proposed method, but with a constant person covariance computed in advance: the steady-state posterior covariance
   of the CV filter updated with positions at 10 Hz with the isotropic LiDAR noise at 3 m plus the localisation error
   (sigma_p ~ 0.039 m, sigma_v ~ 0.105 m/s). Parameters fixed before running, not tuned. Run on main seeds 1000-1099
   (altmain), new seeds 3000-3049 (altrev) and a separate single-process timing campaign (alttiming).
6. **Repeated timing.** In the first timing attempt, run H5_side / seed 3002 / cov took 348 s of wall-clock time while
   other runs took about 4-19 s, with one solve of 8224 ms, consistent with a suspended process (e.g. machine sleep).
   The whole timing campaign was repeated after the alt campaigns with idle sleep disabled; the first attempt is kept as
   `sim_timing_run1.*`. The repetition again contained a few runs above ten times the median wall-clock time; the paper
   therefore reports median percentiles and not the maximum solve time.
7. **Check on DROW v2 (29/10/2026).** The detection rate matched the assumed 0.90 only up to about 3 m and fell to about
   62 % at 6-8 m. Decision of the corresponding author: **keep the simulation unchanged** and report the DROW results
   (Section 5.6 of the paper). No simulation condition was added on the basis of the DROW measurements.
