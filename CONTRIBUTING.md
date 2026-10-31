# Contributing to MentoPlan

Thanks for your interest. Contributions of every size are welcome: bug reports, fixes, documentation,
new import formats, better rendering, and especially **validation data and results**.

## Most valuable contribution: outcome data

The biggest open question is accuracy on real chin-implant patients (see
[`docs/validation.md`](docs/validation.md)). If you have ethically approved, de-identified
pre-/post-operative CT/CBCT or 3-D photographs of chin augmentation or genioplasty (≥ 6 months
post-op), running

```bash
.venv/bin/mentoplan validate-ortho post_op_scan --pre pre_op_scan --out validation/my_case.json
```

and sharing the resulting JSON (numbers only, no images or scans) in an issue is enormously helpful.
**Never upload identifiable patient data to this repository or its issues.**

## Development setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e "backend[dev]"
(cd frontend && npm ci)
scripts/setup_samples.sh                     # optional: public sample CBCTs as demo cases
cd backend && ../.venv/bin/python -m pytest  # integration tests run when a processed case exists
cd frontend && npx vite                      # UI with hot reload (proxies /api to port 8000)
```

Heavy steps (case processing, nonlinear refinement) run in memory-capped child processes; please
keep it that way for new heavy code (`backend/mentoplan/resources.py`).

## Pull requests

* One topic per pull request, with a short description of what and why.
* Add or update tests for behaviour changes (`backend/tests`). Numerical code should come with a
  verification test (analytic solution, patch test, convergence).
* Modelling changes that affect predictions: state the evidence (paper, data) in
  [`docs/evidence.md`](docs/evidence.md) and report the effect on the validation numbers.
* Keep the uncertainty honest: never narrow a range without data that supports it.

## Code style

Python: type hints where helpful, NumPy-style vectorised code, no global state. TypeScript: strict
mode, React function components. Comments explain *why*, not *what*.

By contributing you agree that your contribution is licensed under the MIT License and you follow
the [Code of Conduct](CODE_OF_CONDUCT.md).
