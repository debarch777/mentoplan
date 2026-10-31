"""HTTP API (FastAPI) for the planning client."""
from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..case import CaseStore
from ..implant.design import ImplantDesign
from ..service import PlanningService, design_key
from .bundle import pack

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


class PlanRequest(BaseModel):
    design: dict
    calibrate: bool = True
    uncertainty: str = "validated"
    timepoints: list[tuple[str, float]] | None = None


class DesignRequest(BaseModel):
    design: dict


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def jresp(obj) -> Response:
    return Response(json.dumps(obj, default=_json), media_type="application/json")


def mesh_bundle(mesh, extra_meta=None) -> bytes:
    arrays = {"position": mesh.vertices.astype(np.float32), "index": mesh.faces.astype(np.uint32).ravel(),
              "normal": mesh.vertex_normals().astype(np.float32)}
    for k, v in mesh.attributes.items():
        if not k.startswith("_"):
            arrays["attr_" + k] = np.asarray(v, np.float32)
    return pack(arrays, extra_meta or {})


def create_app(store: CaseStore) -> FastAPI:
    svc = PlanningService(store)
    app = FastAPI(title="MentoPlan", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.svc = svc

    def runtime(cid):
        try:
            rt = svc.runtime(cid)
        except KeyError:
            raise HTTPException(404, f"case {cid} not found")
        if not rt.case.ready:
            raise HTTPException(409, f"case {cid} is not processed yet ({rt.case.meta.get('status')})")
        return rt

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/cases")
    def list_cases():
        return store.list()

    @app.post("/api/cases")
    async def create_case(files: list[UploadFile] = File(...), name: str = Form(...), sex: str = Form(""),
                          age: int | None = Form(None), modality: str | None = Form(None)):
        tmp = Path(tempfile.mkdtemp(prefix="mentoplan_upload_"))
        surf_ext = (".obj", ".ply", ".stl", ".glb", ".gltf")
        surfaces = [f for f in files if Path(f.filename).suffix.lower() in surf_ext]
        aux_ext = (".mtl", ".jpg", ".jpeg", ".png")
        aux = [f for f in files if Path(f.filename).suffix.lower() in aux_ext]
        files = [f for f in files if Path(f.filename).suffix.lower() not in surf_ext + aux_ext]
        surface_path = None
        try:
            (tmp / "scan").mkdir()
            for uf in aux:           # material / texture files referenced by an OBJ
                with (tmp / "scan" / Path(uf.filename).name).open("wb") as f:
                    shutil.copyfileobj(uf.file, f)
            for uf in surfaces[:1]:
                surface_path = tmp / "scan" / Path(uf.filename).name
                with surface_path.open("wb") as f:
                    shutil.copyfileobj(uf.file, f)
            if not files:
                src = None
            elif len(files) == 1:
                dst = tmp / Path(files[0].filename).name
                with dst.open("wb") as f:
                    shutil.copyfileobj(files[0].file, f)
                src = dst
            else:  # a DICOM series uploaded as many files
                src = tmp / "dicom"
                src.mkdir()
                for i, uf in enumerate(files):
                    with (src / f"{i:05d}_{Path(uf.filename).name}").open("wb") as f:
                        shutil.copyfileobj(uf.file, f)
            if src is None and surface_path is None:
                raise HTTPException(422, "upload a CT/CBCT volume and/or a face scan (OBJ/PLY/STL)")
            if src is None and store.default_template() is None:
                raise HTTPException(409, "scan-only planning needs at least one processed CT/CBCT case as template")
            case = store.create(name, src, sex=sex, age=age, modality=modality or None, surface_path=surface_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        svc.submit_processing(case)
        return case.meta

    @app.get("/api/cases/{cid}")
    def get_case(cid: str):
        try:
            case = store.get(cid)
        except KeyError:
            raise HTTPException(404)
        out = {"meta": case.meta}
        if case.ready:
            rt = svc.runtime(cid)
            a = case.anatomy
            out["anatomy"] = {"landmarks": a.landmarks.to_dict()["points"], "notes": a.landmarks.notes,
                              "patient": a.patient, "frame": a.frame.to_dict(),
                              "profiles": {k: v for k, v in a.landmarks.to_dict()["profiles"].items()
                                           if k in ("soft_midline", "bone_symphysis")}}
            out["preop_metrics"] = rt.preop_metrics()
            out["meshes"] = list(case.meshes.keys())
            out["fissure"] = case.model.info.get("fissure")
        return jresp(out)

    @app.get("/api/cases/{cid}/meshes/{name}")
    def get_mesh(cid: str, name: str):
        rt = runtime(cid)
        if name not in rt.case.meshes:
            raise HTTPException(404)
        return Response(mesh_bundle(rt.case.meshes[name]), media_type="application/octet-stream")

    @app.post("/api/cases/{cid}/implant")
    def implant_mesh(cid: str, req: DesignRequest):
        rt = runtime(cid)
        d = ImplantDesign.from_dict(req.design)
        errs = d.validate()
        if errs:
            raise HTTPException(422, "; ".join(errs))
        imp = rt.implant(d)
        return Response(mesh_bundle(imp.mesh, {"metrics": imp.metrics, "design_key": design_key(d)}),
                        media_type="application/octet-stream")

    @app.post("/api/cases/{cid}/implant.stl")
    def implant_stl(cid: str, req: DesignRequest):
        rt = runtime(cid)
        d = ImplantDesign.from_dict(req.design)
        imp = rt.implant(d)
        # manufacturing export in the patient's original scanner coordinates (LPS, mm)
        mesh_w = imp.mesh.transformed(rt.case.anatomy.frame.head_to_world)
        from ..io.meshio import stl_bytes
        name = (d.name or "implant").replace(" ", "_")
        return Response(stl_bytes(mesh_w, header=f"MentoPlan chin implant {name} {design_key(d)}"),
                        media_type="model/stl",
                        headers={"Content-Disposition": f'attachment; filename="{name}.stl"'})

    @app.post("/api/cases/{cid}/plan")
    def plan(cid: str, req: PlanRequest):
        rt = runtime(cid)
        d = ImplantDesign.from_dict(req.design)
        try:
            res = rt.plan(d, req.timepoints, req.calibrate, req.uncertainty)
        except ValueError as e:
            raise HTTPException(422, str(e))
        svc.plans[res["summary"]["plan_id"]] = (cid, res)
        return jresp(res["summary"])

    @app.get("/api/plans/{pid}/fields")
    def plan_fields(pid: str):
        if pid not in svc.plans:
            raise HTTPException(404)
        _, res = svc.plans[pid]
        return Response(pack(res["arrays"], {"n_timepoints": len(res["summary"]["timepoints"])}),
                        media_type="application/octet-stream")

    @app.post("/api/cases/{cid}/refine")
    def refine(cid: str, req: DesignRequest, bg: BackgroundTasks):
        rt = runtime(cid)
        d = ImplantDesign.from_dict(req.design)
        key = design_key(d)
        st = rt.refine_status.get(key)
        if not st or st["state"] == "failed":
            rt.refine_status[key] = {"state": "queued"}
            svc.pool.submit(rt.refine, d)
        return rt.refine_status[key]

    @app.post("/api/cases/{cid}/refine/status")
    def refine_status(cid: str, req: DesignRequest):
        rt = runtime(cid)
        return rt.refine_status.get(design_key(ImplantDesign.from_dict(req.design)), {"state": "none"})

    def motion(rt):
        """Expression / aging fields from the case cache; computed once in a memory-capped child
        process for cases processed before the cache existed."""
        from ..resources import HEAVY_JOB_LOCK, ensure_memory, job_memory_limit_mb, run_isolated
        from ..tissue.motion_cache import load, motion_job
        res = load(rt.case)
        if res is None:
            with HEAVY_JOB_LOCK:
                res = load(rt.case)
                if res is None:
                    try:
                        ensure_memory(2500, "computing expressions")
                        run_isolated(motion_job, str(rt.case.root), limit_mb=job_memory_limit_mb(5000))
                    except Exception as e:
                        raise HTTPException(503, str(e))
                    res = load(rt.case)
        return res

    @app.get("/api/cases/{cid}/expressions")
    def expressions(cid: str):
        ex, _ = motion(runtime(cid))
        arrays = {"active": ex["active"].astype(np.uint32)}
        for k, v in ex["fields"].items():
            arrays["expr_" + k] = v
        return Response(pack(arrays, {"expressions": ex["meta"]}), media_type="application/octet-stream")

    @app.get("/api/cases/{cid}/aging")
    def aging(cid: str):
        _, ag = motion(runtime(cid))
        arrays = {"active": ag["active"].astype(np.uint32)}
        for k, v in ag["fields"].items():
            arrays["age_" + k] = v
        return Response(pack(arrays, {"aging": ag["meta"]}), media_type="application/octet-stream")

    @app.put("/api/cases/{cid}/landmarks")
    def update_landmarks(cid: str, body: dict):
        rt = runtime(cid)
        lms = rt.case.anatomy.landmarks
        for k, v in body.items():
            if k not in lms.points:
                raise HTTPException(422, f"unknown landmark {k}")
            lms.add(k, np.asarray(v, float), 1.0, "manual")
        rt.case.save_anatomy()
        rt.case.rebuild_model()
        rt.reset()
        return {"ok": True, "model": rt.case.model.info}

    @app.get("/api/cases/{cid}/variants")
    def get_variants(cid: str):
        return store.get(cid).variants()

    @app.put("/api/cases/{cid}/variants")
    def put_variants(cid: str, body: list[dict]):
        store.get(cid).save_variants(body)
        return {"ok": True}

    @app.post("/api/cases/{cid}/report", response_class=HTMLResponse)
    def report(cid: str, body: dict):
        rt = runtime(cid)
        from ..report import build_report
        return build_report(rt, body.get("plans", []), body.get("images", {}))

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
    return app
