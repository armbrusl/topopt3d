"""Collect a batch of runs into one machine-readable report.

A long queue job is only useful if what comes back is enough to reason about WITHOUT the machine
that produced it. Grepping logs is not that: the numbers that matter are spread across a `.npz`, a
run log and a reanalysis log, and the failure modes are quiet ones -- a run that stopped early on
its wall-clock budget rather than its iteration count looks exactly like a finished run.

So this reads all three per tag, checks the things that fail silently, and writes both a table for a
human and a JSON for a machine.

    python scripts/cluster_report.py h10_Lagrange h10_cover
    TAGS=h10_Lagrange,h10_cover python scripts/cluster_report.py

Pure numpy -- no jax, no jNO, no pixi environment. It must run anywhere the files land.
"""

import json
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")


def _vols(cells, pts):
    v = pts[cells]
    return np.abs(np.linalg.det(np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], -1))) / 6.0


def _grab(txt, pat, cast=float, default=None):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else default


def _cols(r):
    """Per-iteration history from EITHER npz layout, as {name: 1-D array or None}.

    `bracket.py` and friends write named columns (`it`, `C`, `V`, `mnd`, `P`, `S`, `asp`, ...).
    `lbracket3d.py` designs written before its checkpointing landed carry only a single `hist`
    (N, 4) block of (it, C, V, mnd) and a SCALAR `C`. Reading the first layout's names off the
    second raised `KeyError`, so this report could not open an L-bracket run at all.

    Missing series come back as None rather than zeros -- the printer already renders None as
    "-", and a zero here would read as a measurement.
    """
    names = ("it", "C", "V", "mnd", "P", "S", "ang", "asp", "vmx", "vmn")
    have = {n: np.asarray(r[n]).reshape(-1) for n in names if n in r.files}
    # A scalar `C` beside a `hist` block is the old lbracket layout: `hist` is authoritative.
    if "hist" in r.files and have.get("it") is None and have.get("C", np.zeros(1)).size <= 1:
        h = np.asarray(r["hist"])
        if h.ndim == 2 and h.shape[1] >= 4:
            have.update(dict(zip(("it", "C", "V", "mnd"), [c.reshape(-1) for c in h.T[:4]])))
    return {n: have.get(n) for n in names}


def _last(a):
    return None if a is None or len(a) == 0 else float(a[-1])


def design(tag):
    """Everything the saved design can tell us, plus the checks that fail quietly."""
    p = os.path.join(OUT, f"topo3d_{tag}.npz")
    if not os.path.exists(p):
        return {"tag": tag, "status": "MISSING", "npz": p}
    r = np.load(p)
    cells, p0, pf, rho = r["cells"], r["pts0"], r["pts_f"], r["rho_f"]
    v0, vf = _vols(cells, p0), _vols(cells, pf)
    d = np.linalg.norm(pf - p0, axis=1)
    c = _cols(r)
    it = [int(x) for x in c["it"]] if c["it"] is not None else []
    Ch = [float(x) for x in c["C"]] if c["C"] is not None else []
    # How much the reported compliance depends on WHERE the run stopped. The 2-D conventional arm
    # swings 24-50 % over its last 200 iterations, so its agreement with the paper is an artefact
    # of the stopping iteration; the 3-D arms measure 1-4 %. Any ratio quoted from `C` inherits
    # this as its error bar, so the report computes it rather than leaving it to be forgotten.
    _w = np.asarray(Ch[-8:]) if len(Ch) >= 2 else None
    out = {
        "tag": tag,
        "status": "ok",
        "problem": str(r["problem"]) if "problem" in r.files else "bracket",
        "tets": int(cells.shape[0]),
        "nodes": int(p0.shape[0]),
        "iterations": it[-1] if it else 0,
        "frames": len(it),
        "C": _last(c["C"]),
        "C_history": Ch,
        "C_stoptime_swing_pct": (
            None if _w is None else float(100 * (_w.max() - _w.min()) / _w.mean())
        ),
        "V": _last(c["V"]),
        "M_nd": _last(c["mnd"]),
        "perimeter": _last(c["P"]),
        "smoothness": _last(c["S"]),
        "aspect": _last(c["asp"]),
        "blank_volume_initial": float(v0.sum()),
        "blank_volume_final": float(vf.sum()),
        "material": float((rho * vf).sum()),
        "nodes_moved": int((d > 1e-12).sum()),
        "max_node_move": float(d.max()) if d.size else 0.0,
        "bc_drift_changed": (
            int(np.asarray(r["bc_drift"]).reshape(-1)[2]) if "bc_drift" in r.files else None
        ),
    }
    # --- the quiet failures -------------------------------------------------------------------
    warn = []
    # 1. Stopped on the WALL CLOCK, not the iteration count. bracket.py's loop is
    #    `while time.time() - t0 < BUDGET and it < ITERS`, so a short BUDGET truncates the run and
    #    saves anyway. The saved file is indistinguishable from a finished one.
    log = os.path.join(OUT, f"{tag}.log")
    txt = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    want = _grab(txt, r"ITERS[= ](\d+)", int)
    if want is None:
        want = _grab(txt, r"\((\d+) iterations,", int)
    if want and out["iterations"] < want:
        warn.append(f"stopped at iteration {out['iterations']} of {want} -- BUDGET or a kill, not ITERS")
    out["requested_iters"] = want
    # Independent of any iteration count: a run that finished writes "saved ...". Absent that line,
    # the .npz is a checkpoint from a run that died, and nothing about the target can be recovered
    # from it. This is the check that catches a SIGKILL, where the log has no target to compare to.
    if txt and "saved topo3d_" not in txt:
        warn.append("no completion line in the log -- this .npz is a mid-run checkpoint, not a result")
    if not txt:
        warn.append(f"no run log at {log} -- cannot tell whether this run completed")
    # 2. The design domain moved. Tangential sliding must hold it to machine precision.
    drift = abs(out["blank_volume_final"] / out["blank_volume_initial"] - 1.0)
    if drift > 1e-9:
        warn.append(f"blank volume moved {100 * drift:+.4f} % -- the design domain is not fixed")
    # 3. Supports drifted. bracket.py prints this itself; carry it into the report.
    if "WARNING: the clamped set drifted" in txt:
        warn.append("clamped support set drifted -- reported compliance is not the saved mesh's")
    # 3b. The same fault, from the npz rather than the log. `lbracket3d.py` stamps `bc_drift` into
    #     every checkpoint and then refuses; the design survives, flagged, so the report has to be
    #     the thing that says so -- a stamped-but-unread design is exactly as misleading as an
    #     unstamped one. Absent on designs written before the stamp existed, which is why this is
    #     `is not None` rather than truthiness.
    if out.get("bc_drift_changed") is not None and out["bc_drift_changed"] > 0:
        warn.append(
            f"{out['bc_drift_changed']} boundary node(s) entered or left a BC zone -- the supports "
            f"no longer describe the geometry their predicate names")
    # 3c. The reported compliance depends on where the run stopped. Under ~5 % this is an error
    #     bar; above it, the last iterate is not a result and any ratio built on it inherits that.
    if out.get("C_stoptime_swing_pct") is not None and out["C_stoptime_swing_pct"] > 5.0:
        warn.append(
            f"C swings {out['C_stoptime_swing_pct']:.1f} % over the last frames -- the run is not "
            f"converged and its reported compliance is wherever it happened to stop")
    # 4. A boundary condition sat on movable nodes (jNO's own guard).
    if "anchored to nodes that Variable.trainable()" in txt:
        warn.append("jNO: a boundary condition is anchored to movable nodes")
    out["warnings"] = warn
    out["wall_seconds"] = _grab(txt, r"cpu [\d.]+ s over ([\d.]+) s wall")
    out["cores"] = _grab(txt, r"-> ([\d.]+) cores")
    return out


def _rean_logs(tag):
    """Every reanalysis log for this tag, oldest first.

    The naming diverged: the cluster job writes `reb_<tag>.log`, the deeper hand-run series were
    written as `rean_<tag>_L3fix.log`, and the per-level cluster runs will be `reb_<tag>_L<k>.log`.
    Reading only `reb_<tag>.log` silently reported the SHALLOWEST series available -- measured on
    `hv_Lagrange`, that is a bound of 27.87 where the level-3 series says 111.64, a 4x
    under-report of this repo's headline number, with no warning. Take them all and merge by level.
    """
    pats = (f"reb_{tag}.log", f"reb_{tag}_L*.log", f"rean_{tag}.log", f"rean_{tag}_*.log")
    seen = []
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(OUT, pat))):
            if f not in seen:
                seen.append(f)
    return seen


def reanalysis(tag):
    """The boundary reanalysis, if it ran. Merges every log for the tag, deepest level winning."""
    logs = _rean_logs(tag)
    if not logs:
        return {"status": "not run"}
    txt = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in logs)
    # Levels merge by k across logs; a later log's level k replaces an earlier one's, so a rerun at
    # a deeper level supersedes rather than duplicates.
    lv_by_k = {}
    for f in logs:
        t = open(f, encoding="utf-8", errors="replace").read()
        for a, b, c, x in re.findall(r"level (\d) \(h=([\d.]+),\s+(\d+) tets\): C =\s+([\d.]+)", t):
            lv_by_k[int(a)] = {"k": int(a), "h": float(b), "tets": int(c), "C": float(x)}
    lv = [lv_by_k[k] for k in sorted(lv_by_k)]
    out = {
        "logs": [os.path.basename(f) for f in logs],
        "status": "ok" if lv else "INCOMPLETE",
        "C_design_P1": _grab(txt, r"design, SIMP field, design mesh\s+C =\s+([\d.]+)"),
        "C_kept": _grab(txt, r"extracted solid, the SAME tets\s+C =\s+([\d.]+)"),
        "C_equal_count": _grab(txt, r"EQUAL COUNT, clean tets\s+C =\s+([\d.]+)"),
        "C_opt": _grab(txt, r"C_opt ([\d.]+) ->"),
        "kept_tets": _grab(txt, r"volume-preserving level rho > [\d.]+: (\d+) tets", int),
        "iso_level": _grab(txt, r"volume-preserving level rho > ([\d.]+)"),
        "surface_triangles": _grab(txt, r"extracted surface: (\d+) triangles", int),
        "interior_node_pct": _grab(txt, r"interior \(([\d.]+) ?%\)"),
        "levels": lv,
    }
    if lv:
        Cs = [x["C"] for x in out["levels"]]
        out["truth_lower_bound"] = max(Cs)
        out["monotone"] = all(Cs[i + 1] > Cs[i] for i in range(len(Cs) - 1))
        if len(Cs) > 1:
            out["last_level_change_pct"] = 100 * (Cs[-1] / Cs[-2] - 1)
    return out


def main(tags):
    rep = {}
    for t in tags:
        rep[t] = {"design": design(t), "reanalysis": reanalysis(t)}
        d, ra = rep[t]["design"], rep[t]["reanalysis"]
        # The acceptance test. A LAGRANGE run must reproduce its own reported compliance under an
        # independent P1 solve; anything else means the configuration moved underneath it. This is
        # the check whose absence invalidated a whole set of earlier results.
        if "Lagrange" in t and ra.get("C_opt") and ra.get("C_design_P1"):
            gap = 100 * (ra["C_design_P1"] / ra["C_opt"] - 1)
            ra["acceptance_gap_pct"] = gap
            ra["acceptance"] = "PASS" if abs(gap) < 0.05 else "FAIL"
            if ra["acceptance"] == "FAIL":
                d["warnings"].append(f"acceptance test FAILED: independent solve differs by {gap:+.2f} %")

    dst = os.path.join(OUT, os.environ.get("REPORT", "cluster_report.json"))
    with open(dst, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\n{'tag':<18}{'tets':>8}{'iters':>7}{'C_opt':>10}{'P1 read':>10}{'kept':>8}{'truth >=':>11}{'int%':>7}")
    print("-" * 79)
    for t, v in rep.items():
        d, ra = v["design"], v["reanalysis"]
        if d["status"] != "ok":
            print(f"{t:<18}  MISSING")
            continue
        f = lambda x, w=10, p=4: (f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}")  # noqa: E731
        print(f"{t:<18}{d['tets']:>8}{d['iterations']:>7}{f(d['C'])}{f(ra.get('C_design_P1'))}"
              f"{f(ra.get('kept_tets'), 8, 0)}{f(ra.get('truth_lower_bound'), 11)}"
              f"{f(ra.get('interior_node_pct'), 7, 1)}")
    print()
    for t, v in rep.items():
        ra = v["reanalysis"]
        if ra.get("acceptance"):
            print(f"  {t}: acceptance {ra['acceptance']} ({ra.get('acceptance_gap_pct', float('nan')):+.3f} %)")
        if ra.get("levels") and not ra.get("monotone", True):
            print(f"  {t}: refinement series NOT monotone -- no bound can be read off it")
        elif ra.get("last_level_change_pct") is not None:
            print(f"  {t}: still moving {ra['last_level_change_pct']:+.2f} % at the finest level "
                  f"-- the bound is loose, not a converged value")
        for w in v["design"].get("warnings", []):
            print(f"  {t}: WARNING {w}")
    print(f"\nwrote {dst}")
    return 0 if all(v["design"]["status"] == "ok" for v in rep.values()) else 1


if __name__ == "__main__":
    tags = sys.argv[1:] or [t for t in os.environ.get("TAGS", "").split(",") if t]
    if not tags:
        sys.exit("usage: cluster_report.py TAG [TAG ...]   (or TAGS=a,b)")
    sys.exit(main(tags))
