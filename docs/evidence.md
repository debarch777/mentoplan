# Clinical evidence used as priors

These values are deliberately conservative planning priors; every one of them is configurable in
`backend/mentoplan/tissue/predict.py:ClinicalPriors` and is shown to the user where it influences a
result.

| Quantity | Value used | Source / rationale |
|---|---|---|
| Soft-tissue pogonion advance / implant thickness, by chin-pad thickness | < 7 mm: 0.80; 8–11 mm: 0.66; > 12 mm: 0.50; SD 0.12 | Silicone chin augmentation series (Park; Ko — soft-tissue correction ratio decreasing with baseline soft-tissue thickness). Systematic review of chin procedures: advancement ratio ≈ 0.9 for osseous genioplasty (upper bound for alloplastic implants). |
| Bone resorption / settling at 12 months (mm) | silicone 1.0 ± 0.7, porous PE 0.6 ± 0.5, PEEK 0.4 ± 0.3, titanium 0.3 ± 0.3; scaled with √(projection / 6 mm) | Systematic review of mandibular resorption after chin augmentation: most studies report mean < 2 mm, silicone most variable; older reports 3–5 mm with supraperiosteal/large implants. |
| Resorption time constant | 120 days | Resorption is reported mostly within the first year. |
| Edema | peak 3.5 mm at soft pogonion (log-SD 0.35) at 48–72 h; 65% fast (τ 7 d) + 35% slow (τ 42 d) | Typical clinical course: majority resolved by 4–6 weeks, final contour 3–6 (–12) months. |
| Surgical placement error (SD) | screws: 0.8 mm vertical, 0.6 mm lateral, 1.5° cant; sutures 1.5/1.0/3.0; none 2.5/1.5/4.0 | Planning assumption for a bone-conforming custom implant vs. pocket placement; widened ×1.3 for face-scan-only planning. |
| Model discrepancy | additive 1.16 mm (where change > 1 mm), relative 0.33 | *Calibrated* on the retrospective pre/post surgery pair (`docs/validation.md`) for 90% coverage. |
| Soft-tissue thickness norms (face-scan-only) | pogonion M 12.5 ± 2.0 / F 11.5 ± 2.0 mm; B point 12.0 / 11.0; menton 8.0 / 7.3 | Adult cephalometric soft-tissue thickness data (Holdaway-type analyses). |
| Tissue stiffness | E fat 2.5, muscle 9, dermis 60, mucosa 15 kPa; ν 0.475; log-normal spreads 0.5–0.6 | Ranges used in facial soft-tissue FE models (Chabanas 2003; Barbarino 2009; Mollemans 2007) and in-vivo indentation data. Displacement-driven predictions depend on ratios only. |
| Profile norms | Legan convexity 12 ± 4°, Ricketts E-line (lower lip −2 ± 2 mm, upper −4 ± 2 mm), mentolabial 110–135°, Merrifield Z 80 ± 5°, zero meridian, Sn–Sto : Sto–Me' ≈ 1:2 | Standard soft-tissue cephalometric references. |
| Risk thresholds | mental nerve clearance < 3 mm high, 3–6 moderate; screws ≥ 3–4 mm below root apices; dermal stretch > 25% high; soft-tissue cover < 4 mm high; lengthening > 4 mm or > 12% chin height: mentalis strain | Conservative surgical planning heuristics; shown with their rules in the UI. |

Sources consulted for the chin-specific values:

* Mandibular Bone Resorption Following Chin Augmentation: A Systematic Review — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8989922/
* Soft-tissue response in silicone chin augmentation (Park; Ko), and the systematic review of soft-to-hard tissue ratios of chin procedures — https://pubmed.ncbi.nlm.nih.gov/26321067 , https://e-aaps.org/m/journal/view.php?number=225
