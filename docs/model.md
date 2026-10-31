# How MentoPlan predicts the face

This document describes every step between the patient's scan and the predicted face, the
assumptions made, and how uncertainty is quantified. Numbers marked *calibrated* come from the
validation in `docs/validation.md`; numbers marked *evidence* come from the literature listed in
`docs/evidence.md`.

## 1. From scan to anatomy

| Step | Method | Module |
|---|---|---|
| Read volume | DICOM series (sorted along the slice normal, rescale to HU), NRRD/NHDR, NIfTI, GIPL, MetaImage; LPS world coordinates | `io/volume.py` |
| Intensity calibration | CBCT grey values are not HU. Otsu split air/tissue; air median → −1000, soft-tissue mode → +40 (pseudo-HU) | `imaging/calibrate.py` |
| Non-uniformity correction (CBCT) | Smooth additive bias field from normalised convolution of soft-tissue-range voxels (σ = 12 mm, 3 robust iterations); soft tissue is brought to +40 everywhere. On the two development scans the chin soft tissue read −379 and +367 before correction. | `imaging/segment.py:correct_bias` |
| Skin | Tissue/air boundary at −450; positioning devices removed by 3 mm opening; nostrils and parted lips sealed with a 6 mm closing (concavities of the true surface restored by 4.5 mm recovery); Taubin smoothing removes voxel terracing without shrinkage. Chin rests touching the face: thin (< 7 mm) parts below the chin removed, the skin–device air gap sealed by the closing reopened, then a 2 mm pass removes the device wall left at the contact; landmarks are re-detected on the cleaned skin and the case notes that the submental skin is shown as compressed by the rest. Skin on the cut caps of the scan field of view (padding, empty slices, array edges) is not rendered | `imaging/segment.py`, `case.py`, `viewer/face.ts` |
| Bone | Threshold 5 SD of the soft-tissue noise above soft tissue (≥ 350); nothing within 1.5 mm of the skin; medullary cavities filled | |
| Teeth / metal | Data-driven high-density percentiles inside bone | |
| Head frame | Mid-sagittal plane by *mirror-and-register* (mirror the skeleton + skin, trimmed ICP back onto itself, plane from the −1 eigenvector of the composite improper isometry). RAS axes, pitch from the scanner (CBCT is acquired upright) | `anatomy/frame.py` |
| Landmarks | Mid-sagittal contours of the skin and of the separated mandible; arc-length walks with local extrema and chord-depth criteria. Points on surfaces cut by the scan field of view (convex hull of measured data, array edges) are flagged and never used for measurements. | `anatomy/landmarks.py` |
| Mandible | Watershed on the bone distance map seeded below/above the fitted occlusal surface (splits cusp contacts, brackets and the TMJ) | |
| Mental foramina | Gaps in the buccal cortex (HU sampled along inward normals) in the anatomically plausible band, clustered; population prior with low confidence otherwise | |
| Root apices | Lowest dentin of tooth components connected to the crowns | |
| Arch coordinates | Arc length `u` along the outer arch contour, height `v` above the inferior border | `implant/design.py` |

## 2. Implant design

The implant is a thickness field `t(u, v)` over the external cortex: projection, width, height,
vertical position, peak position, border fullness, under-border wrap, vertical lengthening, outline
(round / anatomical with wings / square), squareness, wing ratio, left/right asymmetry, midline
shift, cant and edge feathering. The solid is built volumetrically (0.35 mm): voxels outside bone,
nearest to the labial/inferior cortex, within `t` of a 1.5 mm-smoothed bone surface. The posterior
surface is therefore an exact negative of the patient's bone (custom fit) and the anterior surface is
smooth. Export: binary STL in the scanner's original coordinates.

## 3. Soft-tissue mechanics

**Mesh.** Voxel hexahedra (2 mm; 3 mm for the whole-face expression model) in the head frame,
classified by 2×2×2 supersampling. Boundary nodes are snapped onto the true skin and bone surfaces
(projected Newton on the smoothed masks), the snap is relaxed into the interior and backed off where
element quality would fall below 0.2 (det J relative to a cube). Cells outside the field of view
below the chin are extrapolated from measured tissue (the neck continues; the scan cut is not a free
surface).

**Materials.** Dermis (outer layer), subcutaneous fat, muscle/glandular tissue, oral mucosa. With
CT the fat is taken from HU; with CBCT (unreliable HU) from the anatomical layering. Median Young's
moduli 2.5 / 9 / 60 / 15 kPa, Poisson 0.475 (*evidence* ranges; only ratios and compressibility
matter for displacement-driven problems).

**Element.** 8-node hexahedron with selective reduced integration: deviatoric terms at 2×2×2 Gauss
points, volumetric term with mean-dilatation (volume-averaged) gradients, which removes volumetric
locking and passes the patch test on distorted elements (`tests/test_hexfem.py`).

**Constitutive models.** Linear elasticity for the interactive solution, the sensitivity analysis and
eigenstrain loads; compressible decoupled neo-Hookean (total Lagrangian, consistent tangent verified
against finite differences, Newton with Dirichlet predictor and line search) for the optional
large-deformation refinement.

**Boundary conditions.**
* bone-attached tissue fixed;
* subperiosteal pocket over the anterior mandible (labial/inferior cortex, never above the root
  apices): the implant prescribes the *normal* displacement `t(u, v)`; tangential sliding is free
  (acute state) — an adhered hypothesis is solved as well and mixed in the ensemble;
* lip mucosa resting on the teeth slides;
* lips split along the oral fissure between the commissures (duplicated nodes, unilateral contact in
  the nonlinear solver);
* ROI cut faces fixed.

## 4. From mechanics to the prediction you see

For each design the engine solves the median model and builds linear fields: implant response,
adhered-interface difference, sensitivities d u / d log θ for fat, muscle, dermis stiffness and the
bulk/shear ratio (direct differentiation with the same factorisation), placement-error responses
(1 mm vertical, 1 mm lateral, 1° cant) and the edema response. A Monte-Carlo ensemble (600 samples)
combines them:

```
u_s(t) = c_s · k_s(t) · [impl + Σ θ_si sens_i + b_s tied + Σ p_sj place_j (+ nonlinear correction)]
         + a_s · h(t) · edema  + ξ_s · σ_add · n
```

* `c_s` — **evidence calibration**: the FE soft:implant ratio at soft pogonion (settled, incl.
  resorption) is fused (inverse-variance) with the clinical ratio for the patient's measured chin-pad
  thickness (0.80 thin, 0.66 normal, 0.50 thick). Times a log-normal *calibrated* model-discrepancy
  factor (relative SD 0.33, minus the part already represented by the evidence spread).
* `k_s(t)` — bone resorption/settling under the implant, material specific, τ = 120 days.
* `a_s h(t)` — edema: volumetric eigenstrain around the dissected pocket and lower lip, magnitude
  log-normal around 3.5 mm at soft pogonion at 48–72 h; resolves ~60% by 2 weeks, ~85% by 6 weeks,
  >95% by 3 months.
* `ξ_s σ_add` — *calibrated* additive discrepancy along the skin normal (1.16 mm where the predicted
  change exceeds 1 mm, fading to the 0.3 mm surface error elsewhere): soft-tissue state/posture and
  segmentation differences between two moments in time.

The client receives the likely surface (ensemble mean) and, per vertex, the normal offsets of the
5th and 95th percentile surfaces; the "uncertainty" slider morphs between them, and every landmark
and measurement carries its 90% interval.

## 5. Expressions and aging

A coarser model of the lower two thirds of the face is driven by Hill-type active fibre stress in
tubes along zygomaticus major, risorius, levator labii superioris, depressor anguli oris, depressor
labii inferioris, mentalis and a circumferential orbicularis oris, and by jaw opening about the
condylar axis (ramus followed to an anatomical Co-Go height when the condyle is not in the scan).
Shapes are patient-specific physics; magnitudes are normalised to typical 3-D motion-capture values
(e.g. 9 mm commissure excursion in a social smile). Aging combines gravity sag under increasing laxity
(scaled to ~1 mm/decade soft-menton descent) and fat-compartment atrophy (3%/decade). Both are
superimposed on the current and the planned face and are labelled as population-average trends.

## 6. Face-scan-only planning

Without CT/CBCT the anatomy is estimated: the scan is framed (up axis, symmetry), its soft landmarks
detected on the mid-sagittal section, and a processed template head is warped onto it with a
thin-plate spline fitted to landmarks, dense non-rigid ICP correspondences and *bone targets* placed
at population soft-tissue thickness (sex specific) below pogonion, B point and menton. Uncertainty is
widened (thickness SD in the ratio prior, larger discrepancy and placement terms) and every estimated
structure is labelled. See `docs/validation.md` for how far this estimate can be from the truth.

## 7. Known limitations

* One retrospective validation pair (orthognathic surgery, not an implant) calibrates the discrepancy
  model. More outcome data will narrow or widen the intervals; the configuration lives in
  `ClinicalPriors`.
* Posture: CBCT is upright, CT supine; supine scans let the chin pad fall back slightly.
* Material parameters are population ranges, not measured per patient.
* Expressions are physically based but not individually calibrated; dynamic wrinkling is not modelled.
* Nerve/root detection must be verified on the scan by the surgeon; the UI shows confidence.
