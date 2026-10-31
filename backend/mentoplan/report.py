"""Self-contained HTML planning report (printable to PDF from the browser)."""
from __future__ import annotations

import html
import time

LIMITATIONS = [
    "Predictions are model estimates with stated 90% ranges, not guarantees. Individual healing varies.",
    "The scan captures the face in one posture and expression; CBCT is usually taken upright, CT supine "
    "(supine scans let the chin pad fall back slightly).",
    "Soft-tissue material properties are not measured for each patient; their uncertainty is part of the range.",
    "Expressions and aging are physics-based illustrations scaled to population averages.",
    "Nerve and root positions are detected automatically; low-confidence detections are flagged and must be "
    "verified on the scan before surgery.",
]


def _f(v, d=1):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "–"


def build_report(rt, plans: list[dict], images: dict) -> str:
    case = rt.case
    a = case.anatomy
    pre = rt.preop_metrics()
    css = """
    body{font:13px/1.45 Inter,system-ui,sans-serif;color:#1b1f24;max-width:960px;margin:24px auto;padding:0 16px}
    h1{font-size:22px;margin:0} h2{font-size:16px;margin:26px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
    table{border-collapse:collapse;width:100%;margin:6px 0} td,th{border-bottom:1px solid #eee;padding:4px 6px;text-align:left}
    th{color:#5a6270;font-weight:500} .mono{font-family:ui-monospace,monospace}
    .imgs{display:flex;gap:8px} .imgs img{width:50%;border-radius:6px}
    .lvl{padding:1px 6px;border-radius:8px;font-size:11px;text-transform:uppercase;font-weight:600}
    .low{background:#dff3e6;color:#1d6b3c}.moderate{background:#fbecd2;color:#8a5a12}.high{background:#f8dad8;color:#8f231c}
    .note{color:#5a6270;font-size:12px} .warn{background:#fbecd2;padding:6px 8px;border-radius:6px;margin:4px 0}
    """
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>Chin implant plan – {html.escape(case.meta['name'])}</title>"
           f"<style>{css}</style></head><body>"]
    out.append(f"<h1>Virtual chin implant plan</h1><div class='note'>{html.escape(case.meta['name'])} · case "
               f"{case.id} · generated {time.strftime('%Y-%m-%d %H:%M')}</div>")
    if images.get("pre") or images.get("post"):
        out.append("<div class='imgs'>")
        for k in ("pre", "post"):
            if images.get(k, "").startswith("data:image/"):
                out.append(f"<img src='{images[k]}' alt='{k}'>")
        out.append("</div>")
    out.append("<h2>Anatomy</h2><table>")
    p = a.patient
    out.append(f"<tr><td>Soft-tissue thickness at pogonion</td><td class='mono'>{_f(p.get('soft_tissue_thickness_pog_mm'))} mm</td></tr>")
    out.append(f"<tr><td>Soft-tissue thickness at menton</td><td class='mono'>{_f(p.get('soft_tissue_thickness_menton_mm'))} mm</td></tr>")
    for k in ("mental_foramen_right", "mental_foramen_left", "anterior_root_apex"):
        if k in a.landmarks:
            lm = a.landmarks.points[k]
            out.append(f"<tr><td>{k.replace('_', ' ')}</td><td class='mono'>{_f(lm.position[0])}, {_f(lm.position[1])}, "
                       f"{_f(lm.position[2])} mm (confidence {_f(lm.confidence, 2)}, {lm.method})</td></tr>")
    out.append("</table>")
    for n in a.landmarks.notes:
        out.append(f"<div class='note'>{html.escape(n)}</div>")
    for plan in plans:
        d = plan["design"]
        out.append(f"<h2>Design: {html.escape(d.get('name', ''))}</h2>")
        out.append("<table><tr>" + "".join(f"<th>{k.replace('_', ' ')}</th>" for k in
                   ("projection", "width", "height", "vertical_extension", "shape", "material", "fixation")) + "</tr><tr>")
        out.append("".join(f"<td class='mono'>{html.escape(str(d.get(k)))}</td>" for k in
                   ("projection", "width", "height", "vertical_extension", "shape", "material", "fixation")) + "</tr></table>")
        im = plan.get("implant", {})
        out.append(f"<div class='note'>Implant volume {_f(im.get('volume_cc'), 2)} cc · max thickness "
                   f"{_f(im.get('max_thickness_mm'))} mm · bone contact {_f(im.get('contact_area_cm2'))} cm²</div>")
        for w in plan.get("warnings", []):
            out.append(f"<div class='warn'>{html.escape(w)}</div>")
        out.append("<h3>Predicted soft-tissue change (median, 90% range), mm</h3><table><tr><th>Time</th>"
                   "<th>Soft pogonion fwd</th><th>Soft menton down</th><th>Lower lip fwd</th><th>Labiomental fold fwd</th></tr>")
        for tp in plan["timepoints"]:
            L = tp["landmarks"]
            def cell(k, i, sign=1):
                if k not in L:
                    return "<td>–</td>"
                x = L[k]
                vals = [sign * (x[w][i] - x["pre"][i]) for w in ("median", "p05", "p95")]
                lo, hi = sorted(vals[1:])
                return f"<td class='mono'>{_f(vals[0])} [{_f(lo)}, {_f(hi)}]</td>"
            out.append(f"<tr><td>{html.escape(tp['label'])}</td>{cell('soft_pogonion', 1)}{cell('soft_menton', 2, -1)}"
                       f"{cell('labrale_inferius', 1)}{cell('soft_b_point', 1)}</tr>")
        out.append("</table>")
        cal = plan["timepoints"][-1].get("calibration", {})
        if "ratio_fem" in cal:
            out.append(f"<div class='note'>Soft:implant ratio – physics model {_f(cal['ratio_fem'], 2)} ± "
                       f"{_f(cal['ratio_fem_sd'], 2)}, clinical evidence {_f(cal['ratio_evidence'], 2)} ± "
                       f"{_f(cal['ratio_evidence_sd'], 2)}, combined {_f(cal['ratio_posterior'], 2)} ± "
                       f"{_f(cal['ratio_posterior_sd'], 2)}.</div>")
        final = plan["timepoints"][-1]
        post = {m["key"]: m for m in final["metrics"]}
        out.append("<h3>Facial balance (settled result)</h3><table><tr><th>Measurement</th><th>Now</th><th>Planned</th><th>Norm</th></tr>")
        for m in pre:
            q = post.get(m["key"])
            rng = f" [{_f(q.get('p05'))}, {_f(q.get('p95'))}]" if q and "p05" in q else ""
            norm = f"{_f(m['norm'][0], 0)}…{_f(m['norm'][1], 0)}" if m["norm"][0] is not None else ""
            out.append(f"<tr><td>{html.escape(m['label'])}</td><td class='mono'>{_f(m['value'])}</td>"
                       f"<td class='mono'>{_f(q['value']) if q else '–'}{rng}</td><td class='mono'>{norm}</td></tr>")
        out.append("</table><h3>Safety assessment</h3><table>")
        for r in plan["risks"]:
            out.append(f"<tr><td><span class='lvl {r['level']}'>{r['level']}</span></td><td><b>{html.escape(r['title'])}</b>"
                       f"<br><span class='note'>{html.escape(r['detail'])}</span></td></tr>")
        out.append("</table>")
    out.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in LIMITATIONS) + "</ul>")
    out.append("<div class='note'>Planning aid for a qualified surgeon; not a substitute for clinical judgement.</div>")
    out.append("</body></html>")
    return "".join(out)
