# MentoPlan user guide

## 1. Import

*Import CT / CBCT / face scan* accepts a DICOM series (select all files or a zip), NRRD, NIfTI, GIPL
or MetaImage volumes, and optionally a face scan (OBJ with its MTL/texture files, PLY, STL).
Processing takes 2–5 minutes: calibration, non-uniformity correction, segmentation, landmarks,
mandible, nerves and the finite-element model. A face scan without CT is planned with estimated
anatomy (an orange banner says so).

Best input: CBCT or CT including the whole chin and submental region, lips relaxed and together,
teeth in occlusion. Parts outside the field of view are detected and extrapolated or excluded.

## 2. Check the anatomy

*Display → Landmarks* shows the detected points (red: mental foramina). The *Analysis* tab lists notes
from the detection (e.g. "nose outside the field of view", "foramen mirrored from the other side").
Low-confidence nerves and roots must be verified on the scan before surgery.

## 3. Design the implant

Start from a preset (central, extended anatomical, square, narrow, vertical lengthening) and adjust:

| Control | Effect |
|---|---|
| Projection | forward augmentation at the peak (mm) |
| Width (arc) | footprint along the jaw incl. wings; wider implants blend into the jaw line |
| Height / Height above border | vertical extent and position on the symphysis |
| Under-border wrap / Vertical lengthening | thin stabilising lip under the border / deliberate chin lengthening |
| Peak position, Border fullness, Squareness, Wings | sagittal and frontal shape |
| Asymmetry, Midline shift, Cant | correct a deviated or canted chin |
| Edge feathering | longer taper = less palpable edges |
| Material, Fixation | affect bone settling, migration and the uncertainty of placement |

Every change re-runs the prediction automatically (about 2–10 s). Use *+* to keep alternative
designs as variants; *Export implant STL* writes the custom implant in the scanner's coordinates as a
starting point for manufacturing (not checked for manufacturability: wall thickness, printability and
sterilisation must be reviewed by the manufacturer); *Nonlinear refine* computes the large-deformation solution (minutes) for the final design.

## 4. Read the prediction

* **Side by side / Wipe / Predicted / Current** and the camera presets show the face from any angle;
  drag to rotate, scroll to zoom.
* **Healing** slider: day 2 to the settled result at one year (swelling, bone settling).
* **Uncertainty** slider: morphs the face between the 5th and 95th percentile outcomes. The
  *validated* profile is calibrated on real surgical outcomes; *relaxed posture* assumes the patient is
  relaxed with lips together before and after.
* **Expression** and **Aging**: smile, broad smile, mouth open, pucker, chin raise, frown; +0 to +20
  years, applied to both the current and the planned face.
* **Analysis** tab: movement of the soft-tissue points with 90% ranges, how the projection estimate was
  formed (physics vs. clinical evidence), and facial-balance measurements now vs. planned against norms.
* **Safety** tab: nerve, roots, bone pressure, migration, skin tension, palpability, lip competence and
  indication, each with its numbers and rule.
* **Display** tab: lighting presets and direction, skin tone, beard shadow, uncertainty / change
  heat-maps, X-ray mode with bone, implant and landmarks.

## 5. Report

*Report* opens a printable summary of the case, all computed variants, ranges, risks and the
limitations of the prediction.
