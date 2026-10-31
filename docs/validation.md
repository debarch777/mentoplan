# Validation

All numbers below are reproducible with the commands shown (data: `scripts/get_sample_data.sh`).

## 1. Finite-element verification (`backend/tests/test_hexfem.py`)

| Test | Result |
|---|---|
| Patch test, regular and randomly distorted hexahedra (ν = 0.49) | linear field reproduced to 1e-9 |
| Consistent tangent vs. central finite differences (distorted element, large strain) | rel. error < 1e-5 |
| Nonlinear residual → linear stiffness at small strain | agrees to 1e-4 |
| Unconfined uniaxial stretch λ = 1.3, nearly incompressible neo-Hookean | reaction force within 0.2% of the analytic solution |
| Cantilever bending, ν = 0.3 vs ν = 0.4999 | no volumetric locking (deflection ratio > 0.75) |

## 2. Retrospective surgical outcome (`mentoplan validate-ortho`)

**Data.** 3D Slicer sample CBCT pair "PreDentalSurgery / PostDentalSurgery": a Class III patient
treated with bimaxillary surgery. Measured by the software: mandibular setback with soft-tissue-
independent chin rotation (bony pogonion −7.0 mm sagittal, −2.5 mm vertical), maxillary advancement
≈ 2 mm.

**Protocol.** Post-op scan registered to the pre-op scan on the cranium (trimmed ICP, 1.1 mm rms).
Bone-segment movements measured by ICP of the pre-op maxilla, mandible and chin segment onto the
post-op bone. The pre-op FE model is driven **only by these bone movements** (soft tissue of the
post-op scan unseen), and the predicted face is compared with the actual post-op face by signed
surface distance along the skin normal over the lower face (14 400 vertices). A whole-face offset
measured where surgery cannot act (forehead, nasal dorsum, cheeks: −0.66 mm, scanner/weight
difference) is removed; raw numbers are given too.

| Lower face, n = 13 500 | Prediction | "No change" baseline |
|---|---|---|
| Mean absolute error | **2.23 mm** | 3.65 mm |
| Median absolute error | 1.73 mm | 3.30 mm |
| RMS | 2.80 mm | 4.50 mm |
| Within 2 mm | **57%** | 33% |
| Within 1 mm | 25% | 18% |
| Raw (no offset correction) MAE | 2.64 mm | — |
| Region that changed > 1 mm (n = 11 000) MAE | 2.51 mm | 4.36 mm |
| 90% prediction band coverage ("validated" profile) | **91%** | — |

Landmark changes (predicted / actual, mm along the surface normal): soft pogonion −6.2 / −13.6, lower
lip −3.4 / −11.8, labiomental fold −2.4 / −5.7, soft menton +0.1 / −3.3.

**Interpretation.** The model removes ~40% of the error of a no-change assumption, and the
predicted soft:hard ratio at pogonion (0.89) matches published values for mandibular setback
(0.9–1.0). The dominant residual — soft pogonion and lower lip retracting about twice as much as the
bone — is physically incompatible with mechanics alone (the chin pad would have to thin from 7.4 to
~1.5 mm). It is the known release of pre-operative lip strain / mentalis activity in Class III patients
(the pre-op scan shows the lips apart over the brackets). This posture change is part of the
"validated" uncertainty profile (additive 2.86 mm, relative 0.36, fitted for 90% coverage), and is
reported separately from the surgical-outcome range in the UI, since a chin implant does not change the
occlusion. The "relaxed posture" profile (additive 1.0 mm) applies when both scans are taken with
relaxed, closed lips.

**Limitations.** One case; jaw surgery rather than an implant (the soft-tissue mechanics are the same,
the boundary condition differs); the two scans come from one scanner at an unknown post-operative
interval (residual swelling possible).

### Comparison with commercial planning software

The most direct published comparison (Knoops et al., Int J Oral Maxillofac Surg 2019; 7 patients,
Le Fort I maxillary osteotomy, 1-year CBCT) reports RMS surface errors of **Dolphin 1.8 ± 0.8 mm,
ProPlan CMF 1.2 ± 0.4 mm and a probabilistic finite-element model 1.3 ± 0.4 mm**. MentoPlan's RMS on
its single available case is 2.8 mm (2.2 mm mean absolute). The cases are not comparable — a
bimaxillary Class III correction whose post-operative scan also differs in lip/mentalis posture (soft
tissue retracting ~2× the bone at the chin, which no bone-driven model can produce) versus isolated
maxillary movements in posture-controlled scans — but **parity with commercial accuracy is not
demonstrated by the data available here**. What is demonstrated: the error is ~40% below a no-change
assumption, the soft:hard ratio of the physics (0.89 at pogonion) matches the setback literature, and the
stated 90% ranges contain 91% of the real outcome. A multi-patient pre/post chin-implant series (CBCT
or 3-D photographs at ≥ 6 months) is the next validation step; `mentoplan validate-ortho` runs on any
such pair unchanged.

## 3. Face-scan-only anatomy estimate (`mentoplan validate-scan`)

**Protocol.** A face scan is simulated from patient A's CT skin (camera-visible surfaces only, random
pose, 0.15 mm noise, synthetic colour). It is planned as a scan-only case with patient B (a
different person) as template, and compared with A's CT anatomy and CT-based prediction.

| Quantity | Error vs. CT |
|---|---|
| Soft-tissue landmarks (rms) | 3.0 mm |
| Bony pogonion / B point / menton | 3.8 / 4.5 / 4.8 mm |
| Mental foramina (right / left) | 3.6 / 1.4 mm |
| Anterior root apices | 12.6 mm (not estimable from a face scan) |
| Chin-pad thickness | 12.4 mm estimated (population norm) vs 7.4 mm true |

6 mm implant, soft pogonion advance at 1 year: **CT 3.1 mm [1.6, 5.9]**, **face scan only 2.1 mm
[1.0, 4.7]** — the CT-based median lies inside the scan-only range, the scan-only estimate is lower
because this patient's chin pad is unusually thin (an outlier against the population norm). Conclusion:
scan-only planning is useful for visualising options; CT/CBCT is required to plan surgery (nerve, roots,
pad thickness).

## 4. Pipeline integration tests (`backend/tests/test_integration.py`)

Zero implant → zero change; monotonic response to projection with physically plausible soft:hard
ratios (0.3–1.1); symmetric design → no midline drift; asymmetric design → larger change on the
augmented side; edema at one week > settled result; 5th–95th band brackets the median and never implies
a retruded chin; titanium (less resorption) > silicone at one year; risk levels increase with implant
size; custom implant solid is watertight with the requested thickness; HTTP API smoke test (meshes,
plan validation, STL).
