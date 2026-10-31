"""Command line interface.

    mentoplan ingest <volume> --name NAME [--sex F|M] [--age N] [--modality CT|CBCT]
    mentoplan process <case_id>
    mentoplan list
    mentoplan serve [--host 127.0.0.1] [--port 8000]
    mentoplan validate-ortho <post_volume> (--case <pre_case_id> | --pre <pre_volume>)
    mentoplan validate-scan <subject_case_id> <template_case_id>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("MENTOPLAN_DATA", Path(__file__).resolve().parents[2] / "data" / "cases"))


def _process_isolated(store, case):
    """Process a case in a memory-capped child process (as the server does), so a large scan fails
    with a message instead of exhausting the machine."""
    from .resources import ensure_memory, job_memory_limit_mb, run_isolated
    from .service import PROCESS_LIMIT_MB, PROCESS_MIN_MB, process_job
    ensure_memory(PROCESS_MIN_MB, "processing a scan")
    tmpl = None if case.meta.get("volume_input") else store.default_template()
    run_isolated(process_job, str(case.root), str(tmpl.root) if tmpl else None,
                 limit_mb=job_memory_limit_mb(PROCESS_LIMIT_MB))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mentoplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest", help="create a case from a CT/CBCT volume and process it")
    a.add_argument("volume", nargs="?", help="CT/CBCT (DICOM dir/zip, NRRD, NIfTI, GIPL, MHA); omit for scan-only")
    a.add_argument("--name", required=True)
    a.add_argument("--sex", default="")
    a.add_argument("--age", type=int, default=None)
    a.add_argument("--modality", default=None)
    a.add_argument("--surface", default=None, help="optional textured surface scan (OBJ/PLY)")
    a.add_argument("--no-process", action="store_true")
    p = sub.add_parser("process")
    p.add_argument("case_id")
    sub.add_parser("list")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    v = sub.add_parser("validate-ortho", help="retrospective soft-tissue prediction on a pre/post surgery pair")
    v.add_argument("post", help="post-operative CT/CBCT")
    v.add_argument("--pre", help="pre-operative CT/CBCT (a new case is processed)")
    v.add_argument("--case", help="already processed pre-operative case id")
    v.add_argument("--out", default="validation/ortho.json")
    w = sub.add_parser("validate-scan", help="face-scan-only planning vs. the CT of the same person")
    w.add_argument("subject_case")
    w.add_argument("template_case")
    w.add_argument("--out", default="validation/scan_only.json")
    args = ap.parse_args(argv)

    from .resources import cap_threads
    cap_threads()
    from .case import CaseStore
    store = CaseStore(data_root())
    if args.cmd == "ingest":
        case = store.create(args.name, Path(args.volume) if args.volume else None, sex=args.sex, age=args.age,
                            modality=args.modality, surface_path=Path(args.surface) if args.surface else None)
        print(f"created case {case.id}")
        if not args.no_process:
            _process_isolated(store, case)
    elif args.cmd == "process":
        _process_isolated(store, store.get(args.case_id))
    elif args.cmd == "list":
        for c in store.list():
            print(c["id"], c["name"], c.get("status", {}).get("stage"))
    elif args.cmd == "serve":
        import uvicorn
        from .api.server import create_app
        uvicorn.run(create_app(store), host=args.host, port=args.port)
    elif args.cmd == "validate-ortho":
        from .validation.ortho import run_validation
        case = store.get(args.case) if args.case else None
        run_validation(Path(args.pre) if args.pre else None, Path(args.post), Path(args.out), case=case)
    elif args.cmd == "validate-scan":
        from .validation.scan_only import run
        run(store, args.subject_case, args.template_case, Path(args.out))


if __name__ == "__main__":
    main()
