"""Chiến dịch mô phỏng cho nghiên cứu mô phỏng (phương án A).

python3 tools/sim_campaign.py --campaign dev       # hạt 0–49: kiểm tra, không báo cáo
python3 tools/sim_campaign.py --campaign main      # hạt 1000–1099, 3 phương pháp × 5 kịch bản
python3 tools/sim_campaign.py --campaign robust    # hạt 2000–2049: nhiễu ×2, trễ ×1.5, người né robot
python3 tools/sim_campaign.py --campaign ablation  # hạt 1000–1099, biến thể của phương pháp cov
Kết quả: results/sim_<campaign>.csv (ghi tiếp được khi bị ngắt) và results/sim_<campaign>.meta.json.
"""
import argparse
import hashlib
import itertools
import json
import os
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS, ROOT, code_hash  # noqa: E402

from pu_nmpc_core import config  # noqa: E402
from pu_nmpc_core.constraint import steady_state_cov  # noqa: E402
from pu_nmpc_core.controller import Controller, MethodParams  # noqa: E402
from pu_nmpc_core.models import RobotParams  # noqa: E402
from pu_nmpc_core.simstudy import SCENARIOS, SensorSim, StudyConfig, run_episode  # noqa: E402
from pu_nmpc_core.tracker import SensorModel  # noqa: E402

SS = ROOT / "config" / "simstudy.yaml"


def v(d, k):
    return d[k]["value"]


def lidar_sigma(r, c):
    acc = np.where(r <= 3, 0.01, np.where(r <= 5, 0.02, 0.025))
    return np.sqrt(c ** 2 + (acc * r) ** 2)


def build():
    s = yaml.safe_load(SS.read_text())
    ro, li, ca = s["robot"], s["lidar"], s["camera"]
    r_w = v(ro, "wheel_radius")
    rp = RobotParams(wheel_radius=r_w, lx=v(ro, "lx"), ly=v(ro, "ly"), wheel_speed_max=v(ro, "v_max") / r_w,
                     body_acc_max=tuple(v(ro, "body_acc_max")), tau=tuple(v(ro, "tau_model")), radius=v(ro, "radius"),
                     omega_max=v(ro, "omega_max"))
    # mô hình nhiễu tuyến tính a + b·r khớp với mô hình datasheet/giả định trên dải dùng
    rl = np.linspace(0.3, v(li, "range_max_used"), 200)
    bl, al = np.polyfit(rl, lidar_sigma(rl, v(li, "centroid_std")), 1)
    rc = np.linspace(*v(ca, "range"), 200)
    k = v(ca, "depth_std_at_2m") / 4.0
    bc, ac = np.polyfit(rc, k * rc ** 2, 1)
    ac = max(ac, 0.005)
    lidar = SensorModel("lidar", float(al), float(bl), np.deg2rad(v(li, "bearing_std_deg")), None, v(li, "range_max_used"))
    cam = SensorModel("camera", float(ac), float(bc), np.deg2rad(v(ca, "bearing_std_deg")), np.deg2rad(v(ca, "fov_h_deg") / 2),
                      v(ca, "range")[1], mount=(0.25, 0.0, 0.0))
    sensors = dict(lidar=SensorSim(lidar, v(li, "rate"), v(li, "delay"), v(li, "p_detect"), v(li, "clutter_per_scan")),
                   camera=SensorSim(cam, v(ca, "rate_detector"), v(ca, "delay"), v(ca, "p_detect"), v(ca, "clutter_per_scan")))
    cfg = StudyConfig(reactive=True, actuation_delay=v(ro, "actuation_delay"), tau_true_range=tuple(v(ro, "tau_true_range")),
                      loc_std=tuple(v(s["localization"], "std")), footprint=tuple(v(ro, "footprint")))
    mp, npar, refp = config.controller(ROOT / "config" / "controller.yaml")
    tp = config.tracker(ROOT / "config" / "tracker.yaml")
    q = v(s["identified"], "q_acc")
    tp.q_acc = q
    mp.q_acc_pred = q
    # đối chứng "ccconst": hiệp phương sai dừng của bộ lọc khi LiDAR thấy người liên tục ở 3 m (chốt trước khi chạy)
    r0, ls = 3.0, v(s["localization"], "std")
    r_var = 0.5 * (lidar.covariance_polar(r0)[0, 0] + r0 ** 2 * lidar.covariance_polar(r0)[1, 1]) + ls[0] ** 2 + 0.5 * (r0 * ls[2]) ** 2
    mp.const_cov = tuple(steady_state_cov(1.0 / v(li, "rate"), q, r_var).ravel())
    npar.input_delay = v(ro, "actuation_delay")     # bộ điều khiển biết trễ chấp hành (như sau bước nhận dạng P3)
    return s, rp, sensors, cfg, mp, npar, refp, tp, dict(lidar_ab=(float(al), float(bl)), camera_ab=(float(ac), float(bc)))


CONDITIONS = {
    # tên: (phương pháp, thay đổi cấu hình)
    "nominal": {},
    "noise2": dict(noise_scale=2.0),
    "delay15": dict(delay_scale=1.5),
    "nonreactive": dict(reactive=False),
    "nocomp": dict(compensate=False),
    "lidar_only": dict(sensors_enabled=("lidar",)),
    "cantelli": dict(kappa_kind="cantelli"),
    # bản sửa sau phản biện (hạt mới 3000–3049, chưa từng chạy)
    "covcal": dict(cov_inflate="calibrated"),   # γ đọc từ simstudy.yaml: revision.cov_inflate (hiệu chỉnh trên hạt dev)
    "fix045": dict(d_fixed=0.45),
    "fix060": dict(d_fixed=0.60),
    "fix075": dict(d_fixed=0.75),
}
CAMPAIGNS = {
    "dev": dict(seeds=range(0, 10), methods=["cov", "fixed", "heuristic"], conds=["nominal"]),
    "main": dict(seeds=range(1000, 1100), methods=["cov", "fixed", "heuristic"], conds=["nominal"]),
    "robust": dict(seeds=range(2000, 2050), methods=["cov", "fixed", "heuristic"], conds=["noise2", "delay15", "nonreactive"]),
    "ablation": dict(seeds=range(1000, 1100), methods=["cov"], conds=["nocomp", "lidar_only", "cantelli"]),
    "devcal": dict(seeds=range(0, 10), methods=["cov"], conds=["nominal"]),
    "revision": dict(seeds=range(3000, 3050), methods=["cov", "fixed", "heuristic"], conds=["nominal"],
                     extra=[("cov", "covcal"), ("fixed", "fix045"), ("fixed", "fix060"), ("fixed", "fix075")]),
    "altmain": dict(seeds=range(1000, 1100), methods=["ccconst"], conds=["nominal"]),
    "altrev": dict(seeds=range(3000, 3050), methods=["ccconst"], conds=["nominal"]),
    "alttiming": dict(seeds=range(3000, 3010), methods=["ccconst"], conds=["nominal"]),
    "timing": dict(seeds=range(3000, 3010), methods=["cov", "fixed", "heuristic"], conds=["nominal"]),
}

_W = {}


def _init():
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    s, rp, sensors, cfg, mp, npar, refp, tp, _ = build()
    rv = s.get("revision", {}).get("cov_inflate")
    _W["cov_inflate"] = float(rv["value"]) if rv else None
    _W.update(rp=rp, sensors=sensors, cfg=cfg, mp=mp, npar=npar, refp=refp, tp=tp,
              ctrl=Controller(rp, MethodParams(**mp.__dict__), npar))


def _job(args):
    scen, seed, method, cond = args
    from dataclasses import replace
    import copy
    ch = CONDITIONS[cond]
    mp = replace(_W["mp"], method=method)
    if "kappa_kind" in ch:
        mp = replace(mp, kappa_kind=ch["kappa_kind"])
    if "d_fixed" in ch:
        mp = replace(mp, d_fixed=ch["d_fixed"])
    if "cov_inflate" in ch:
        mp = replace(mp, cov_inflate=_W["cov_inflate"])
    cfg = replace(_W["cfg"], **{k: ch[k] for k in ("reactive", "compensate", "sensors_enabled", "delay_scale") if k in ch})
    sensors = copy.deepcopy(_W["sensors"])
    if "noise_scale" in ch:
        for s in sensors.values():
            s.noise_scale = ch["noise_scale"]
    t0 = time.time()
    try:
        r = run_episode(scen, seed, mp, _W["rp"], _W["npar"], _W["refp"], _W["tp"], sensors, cfg, ctrl=_W["ctrl"])
    except Exception as e:  # noqa: BLE001
        r = dict(scenario=scen, seed=seed, method=method, outcome="error", error=repr(e)[:200])
    r.update(condition=cond, wall_s=time.time() - t0)
    return r


def main(name, workers):
    camp = CAMPAIGNS[name]
    out = RESULTS / f"sim_{name}.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        done = set(zip(d.scenario, d.seed, d.method, d.condition))
    allj = list(itertools.product(SCENARIOS, camp["seeds"], camp["methods"], camp["conds"]))
    allj += [(sc, sd, m, c) for sc in SCENARIOS for sd in camp["seeds"] for m, c in camp.get("extra", [])]
    jobs = [j for j in allj if j not in done]
    s, rp, sensors, cfg, mp, npar, refp, tp, fit = build()
    meta = dict(kind="simulation", synthetic=False, campaign=name, code=code_hash(),
                simstudy_sha=hashlib.sha256(SS.read_bytes()).hexdigest()[:16],
                controller_sha=hashlib.sha256((ROOT / "config" / "controller.yaml").read_bytes()).hexdigest()[:16],
                tracker_sha=hashlib.sha256((ROOT / "config" / "tracker.yaml").read_bytes()).hexdigest()[:16],
                sensor_fit=fit, const_cov=list(mp.const_cov), n_jobs_total=len(jobs) + len(done), machine=os.uname().machine,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (RESULTS / f"sim_{name}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {len(jobs)} lượt cần chạy ({len(done)} đã có), {workers} tiến trình", flush=True)
    t0, k = time.time(), 0
    with Pool(workers, initializer=_init) as pool:
        for r in pool.imap_unordered(_job, jobs, chunksize=2):
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            k += 1
            if k % 50 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(jobs)}  {el / k * (len(jobs) - k) / 60:.1f} phút còn lại", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RESULTS / f"sim_{name}.meta.json").write_text(json.dumps(meta, indent=1))
    print("XONG", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=list(CAMPAIGNS), required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    main(a.campaign, a.workers)
