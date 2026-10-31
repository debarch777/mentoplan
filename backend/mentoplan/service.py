"""Planning service: per-case runtime state, caching and result composition."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .analysis import cephalometry as ceph
from .analysis import risk as riskmod
from .case import Case, CaseStore
from .implant.design import ImplantDesign, build_implant
from .resources import HEAVY_JOB_LOCK, ensure_memory, job_memory_limit_mb, run_isolated
from .tissue.predict import TIMEPOINTS, PredictionEngine

# memory budgets (MB) for heavy jobs; the child process is hard-limited, the server stays responsive
PROCESS_MIN_MB, PROCESS_LIMIT_MB = 3500, 7000
REFINE_MIN_MB, REFINE_LIMIT_MB = 3000, 5000


def process_job(case_root: str, template_root: str | None):
    """Child-process entry: full case processing (segmentation ... FE model)."""
    from pathlib import Path
    case = Case(Path(case_root))
    template = Case(Path(template_root)) if template_root else None
    case.process(template=template)
    if case.meta.get("volume_input") and case.meta.get("surface_input"):
        from .surface_scan import attach_surface_scan
        attach_surface_scan(case)
    return True


def refine_job(case_root: str, design: dict):
    """Child-process entry: nonlinear (neo-Hookean) solution of one design; returns nodal displacements."""
    from pathlib import Path
    from .tissue.simulate import Simulator
    case = Case(Path(case_root))
    sim = Simulator(case.model)
    d = ImplantDesign.from_dict(design)
    from .implant.design import node_thickness
    g = node_thickness(case.model.nodes[sim.pocket], sim.pocket_normals, case.anatomy.arch, d)
    u_nl, _ = sim.solve_nonlinear(g, interface="sliding", n_steps=2, tol=1e-4)
    return u_nl.astype(np.float64)

UNCHANGED = ("soft_glabella", "soft_nasion", "pronasale")


def design_key(d: ImplantDesign) -> str:
    dd = d.to_dict()
    dd.pop("name", None)
    return hashlib.sha1(json.dumps(dd, sort_keys=True).encode()).hexdigest()[:16]


class LRU(OrderedDict):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def get_or(self, k, fn):
        if k in self:
            self.move_to_end(k)
            return self[k]
        v = fn()
        self[k] = v
        while len(self) > self.n:
            self.popitem(last=False)
        return v


class CaseRuntime:
    def __init__(self, case: Case):
        self.case = case
        self.lock = threading.RLock()
        self._engine: PredictionEngine | None = None
        self.solutions = LRU(8)
        self.implants = LRU(16)
        self.refine_status: dict[str, dict] = {}

    @property
    def engine(self) -> PredictionEngine:
        with self.lock:
            if self._engine is None:
                self._engine = PredictionEngine(self.case)
            return self._engine

    def reset(self):
        with self.lock:
            if self._engine is not None:          # release solver (MKL) memory explicitly
                for ls in self._engine.sim._systems.values():
                    if ls._solver is not None:
                        ls._solver.free()
            self._engine = None
            self.solutions.clear()
            self.implants.clear()

    # ------------------------------------------------------------------ geometry
    def bonefield(self):
        """Smoothed bone fields for implant construction; the full segmentation is released after
        they are built (planning never needs it again - saves ~0.5 GB per open case)."""
        if getattr(self, "_bf", None) is None:
            self._bf = self.case.bonefield
            self._bf.release_segmentation()
            self.case._seg = None
        return self._bf

    def implant(self, d: ImplantDesign):
        key = design_key(d)
        a = self.case.anatomy
        bf = self.bonefield()
        return self.implants.get_or(key, lambda: build_implant(d, a.arch, bf.sample_mandible, a.landmarks))

    def solution(self, d: ImplantDesign):
        key = design_key(d)
        with self.lock:
            return self.solutions.get_or(key, lambda: self.engine.solve_design(d))

    # ------------------------------------------------------------------ metrics
    def landmark_dict(self, positions: dict | None = None) -> dict:
        """Landmarks usable for measurements (low-confidence ones, e.g. outside the field of view,
        are left out so that no metric is computed from them)."""
        lms = self.case.anatomy.landmarks
        out = {k: lms[k] for k, v in lms.points.items() if v.confidence >= 0.3 or v.method == "manual"}
        if positions:
            out.update(positions)
        return out

    def preop_metrics(self) -> list[dict]:
        L = self.landmark_dict()
        m = ceph.profile_metrics(L)
        w = ceph.chin_width(self.case.meshes["skin"].vertices, L["soft_pogonion"])
        if w is not None:
            m["chin_width"] = w
        return ceph.describe(m)

    def _metric_intervals(self, lm_samples: dict, disp=None) -> list[dict]:
        names = list(lm_samples)
        S = len(next(iter(lm_samples.values())))
        base = self.landmark_dict()
        for n in getattr(self._engine, "unavailable", []):
            base.pop(n, None)               # no skin support -> metrics using it are not reported
        vals: dict[str, list] = {}
        for s in range(S):
            L = dict(base)
            for n in names:
                L[n] = lm_samples[n][s]
            for k, v in ceph.profile_metrics(L).items():
                vals.setdefault(k, []).append(v)
        med = {k: float(np.median(v)) for k, v in vals.items()}
        if disp is not None:
            skin = self.case.meshes["skin"].vertices
            w = ceph.chin_width(skin + disp, base["soft_pogonion"] + np.median(lm_samples["soft_pogonion"], 0)
                                - base["soft_pogonion"])
            if w is not None:
                med["chin_width"] = w
        rows = ceph.describe(med)
        for r in rows:
            if r["key"] in vals:
                r["p05"] = float(np.percentile(vals[r["key"]], 5))
                r["p95"] = float(np.percentile(vals[r["key"]], 95))
        return rows

    # ------------------------------------------------------------------ planning
    def plan(self, d: ImplantDesign, timepoints=None, calibrate: bool = True, uncertainty: str = "validated") -> dict:
        t0 = time.time()
        errs = d.validate()
        if errs:
            raise ValueError("; ".join(errs))
        sol = self.solution(d)
        imp = self.implant(d)
        eng = self.engine
        eng.priors.use_profile(uncertainty)
        tps = timepoints or TIMEPOINTS
        results = []
        arrays = {"active": eng.active.astype(np.uint32)}
        with self.lock:
            for i, (label, t) in enumerate(tps):
                r = eng.predict(sol, t, calibrate=calibrate)
                act = eng.active
                arrays[f"disp_{i}"] = r["disp"][act]
                arrays[f"low_{i}"] = r["low"][act]
                arrays[f"high_{i}"] = r["high"][act]
                lm_samples = {k: v["samples"] for k, v in r["landmarks"].items()}
                metrics = self._metric_intervals(lm_samples, r["disp"])
                lm_json = {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in r["landmarks"].items()}
                results.append({"label": label, "t_days": t, "swelling_fraction": r["swelling_fraction"],
                                "landmarks": lm_json, "metrics": metrics, "calibration": r["calibration"],
                                "posture_sd_mm": r["posture_sd_mm"]})
            final = results[-1]
            ls = eng.sim.system("sliding")
            mand = self.case.meshes["mandible"]
            risks = riskmod.assess(d, self.case.anatomy, self.case.model, sol.fields,
                                   mand.vertices, imp.mesh, imp.metrics,
                                   final["landmarks"], ls.k_pen, eng.pocket, eng.sim.pocket_normals,
                                   eng.pocket_thickness(d), mand_normals=mand.vertex_normals())
        plan_id = uuid.uuid4().hex[:12]
        summary = {
            "plan_id": plan_id, "design": d.to_dict(), "design_key": design_key(d),
            "timepoints": results, "risks": risks, "overall_risk": riskmod.overall(risks),
            "implant": imp.metrics, "warnings": sol.warnings, "nonlinear": sol.nonlinear,
            "pocket_coverage": sol.pocket_coverage, "t_pog": sol.t_pog,
            "compute_seconds": round(time.time() - t0, 2), "uncertainty_profile": uncertainty,
        }
        return {"summary": summary, "arrays": arrays}

    def refine(self, d: ImplantDesign, log=None):
        """Large-deformation refinement in a memory-capped child process (one heavy job at a time)."""
        key = design_key(d)
        self.refine_status[key] = {"state": "queued"}
        try:
            sol = self.solution(d)
            with HEAVY_JOB_LOCK:
                ensure_memory(REFINE_MIN_MB, "the nonlinear refinement")
                self.refine_status[key] = {"state": "running", "started": time.time()}
                u_nl = run_isolated(refine_job, str(self.case.root), d.to_dict(),
                                    limit_mb=job_memory_limit_mb(REFINE_LIMIT_MB))
            with self.lock:
                sol.fields["nl"] = u_nl - sol.fields["impl"]
                sol.nonlinear = True
                sol._vf = None
            self.refine_status[key] = {"state": "done", "finished": time.time()}
        except Exception as e:  # reported to the client
            self.refine_status[key] = {"state": "failed", "error": str(e)}


class PlanningService:
    def __init__(self, store: CaseStore):
        self.store = store
        self.runtimes: dict[str, CaseRuntime] = {}   # insertion order = LRU order
        self.plans = LRU(24)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    MAX_OPEN_CASES = 2      # each open case holds ~2.5-3 GB (factorisation, fields, bone volumes)

    def runtime(self, cid: str) -> CaseRuntime:
        with self._lock:
            if cid in self.runtimes:
                rt = self.runtimes.pop(cid)
            else:
                rt = CaseRuntime(self.store.get(cid))
            self.runtimes[cid] = rt                      # most recently used last
            while len(self.runtimes) > self.MAX_OPEN_CASES:
                old_id = next(iter(self.runtimes))
                old = self.runtimes.pop(old_id)
                with old.lock:
                    old.reset()
                    old._bf = None
                import gc
                gc.collect()
            return rt

    def submit_processing(self, case: Case):
        """Process a new case in a memory-capped child process, one heavy job at a time."""
        def run():
            try:
                with HEAVY_JOB_LOCK:
                    ensure_memory(PROCESS_MIN_MB, "processing a scan")
                    tmpl = None if case.meta.get("volume_input") else self.store.default_template()
                    run_isolated(process_job, str(case.root), str(tmpl.root) if tmpl else None,
                                 limit_mb=job_memory_limit_mb(PROCESS_LIMIT_MB))
            except Exception as e:
                Case(case.root).set_status("failed", 1.0, f"{type(e).__name__}: {e}")
        return self.pool.submit(run)
