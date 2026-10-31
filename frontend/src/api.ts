// Backend API client and binary bundle decoding.

export type Vec3 = [number, number, number]

export interface Landmark {
  name: string
  position: Vec3
  confidence: number
  method: string
}

export interface MetricRow {
  key: string
  label: string
  value: number
  unit: string
  norm: [number | null, number | null]
  status: string
  depends_on_head_pose: boolean
  reference: string
  p05?: number
  p95?: number
}

export interface CaseMeta {
  id: string
  name: string
  created: number
  sex?: string
  age?: number | null
  modality?: string | null
  status: { stage: string; progress: number; message: string }
  model_info?: Record<string, unknown>
  anatomy_source?: string
  surface_registration?: { rms_mm: number; coverage: number }
}

export interface CaseDetail {
  meta: CaseMeta
  anatomy?: {
    landmarks: Record<string, Landmark>
    notes: string[]
    patient: Record<string, unknown>
    profiles: Record<string, [number, number][]>
  }
  preop_metrics?: MetricRow[]
  meshes?: string[]
  fissure?: { z: number; half_width_mm: number } | string
}

export interface ImplantDesign {
  projection: number
  width: number
  height: number
  inferior_offset: number
  vertical_extension: number
  peak_height: number
  base_fill: number
  inferior_wrap: number
  shape: 'round' | 'anatomical' | 'square' | 'custom'
  squareness: number
  wing_ratio: number
  body_fraction: number
  asymmetry: number
  midline_shift: number
  cant_deg: number
  edge_feather: number
  material: 'silicone' | 'porous_polyethylene' | 'peek' | 'titanium'
  fixation: 'screws' | 'sutures' | 'none'
  name: string
}

export const DEFAULT_DESIGN: ImplantDesign = {
  projection: 6, width: 50, height: 18, inferior_offset: 0, vertical_extension: 0, peak_height: 0.45,
  base_fill: 0.75, inferior_wrap: 1.5, shape: 'anatomical', squareness: 0.3, wing_ratio: 0.35, body_fraction: 0.5, asymmetry: 0,
  midline_shift: 0, cant_deg: 0, edge_feather: 6, material: 'silicone', fixation: 'screws', name: 'Variant A',
}

export interface LandmarkPrediction {
  pre: Vec3
  median: Vec3
  p05: Vec3
  p95: Vec3
}

export interface TimepointResult {
  label: string
  t_days: number
  swelling_fraction: number
  landmarks: Record<string, LandmarkPrediction>
  metrics: MetricRow[]
  calibration: Record<string, number | boolean>
  posture_sd_mm: number
}

export interface Risk {
  id: string
  category: string
  title: string
  level: 'low' | 'moderate' | 'high'
  value: number
  unit: string
  detail: string
  rule: string
}

export interface PlanSummary {
  plan_id: string
  design: ImplantDesign
  design_key: string
  timepoints: TimepointResult[]
  risks: Risk[]
  overall_risk: string
  implant: { volume_cc: number; max_thickness_mm: number; contact_area_cm2: number }
  warnings: string[]
  nonlinear: boolean
  pocket_coverage: number
  t_pog: number
  compute_seconds: number
}

export type Arrays = Record<string, Float32Array | Uint32Array | Int32Array | Uint8Array | Uint16Array>

export interface Bundle {
  meta: Record<string, any>
  arrays: Arrays
  shapes: Record<string, number[]>
}

const CTORS: Record<string, any> = { f32: Float32Array, u32: Uint32Array, i32: Int32Array, u8: Uint8Array, u16: Uint16Array }

export function decodeBundle(buf: ArrayBuffer): Bundle {
  const dv = new DataView(buf)
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3))
  if (magic !== 'MPB1') throw new Error('bad bundle')
  const hl = dv.getUint32(4, true)
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)))
  const base = 8 + hl
  const arrays: Arrays = {}
  const shapes: Record<string, number[]> = {}
  for (const e of header.arrays) {
    const C = CTORS[e.dtype]
    const n = e.shape.reduce((a: number, b: number) => a * b, 1)
    arrays[e.name] = new C(buf.slice(base + e.offset, base + e.offset + e.nbytes), 0, n)
    shapes[e.name] = e.shape
  }
  return { meta: header.meta, arrays, shapes }
}

async function check(r: Response) {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const j = await r.json()
      msg = j.detail || msg
    } catch { /* ignore */ }
    throw new Error(msg)
  }
  return r
}

export const api = {
  async listCases(): Promise<CaseMeta[]> {
    return (await check(await fetch('/api/cases'))).json()
  },
  async getCase(id: string): Promise<CaseDetail> {
    return (await check(await fetch(`/api/cases/${id}`))).json()
  },
  async createCase(form: FormData): Promise<CaseMeta> {
    return (await check(await fetch('/api/cases', { method: 'POST', body: form }))).json()
  },
  async mesh(id: string, name: string): Promise<Bundle> {
    return decodeBundle(await (await check(await fetch(`/api/cases/${id}/meshes/${name}`))).arrayBuffer())
  },
  async implant(id: string, design: ImplantDesign): Promise<Bundle> {
    const r = await check(await fetch(`/api/cases/${id}/implant`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ design }),
    }))
    return decodeBundle(await r.arrayBuffer())
  },
  async implantStl(id: string, design: ImplantDesign): Promise<Blob> {
    const r = await check(await fetch(`/api/cases/${id}/implant.stl`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ design }),
    }))
    return r.blob()
  },
  async plan(id: string, design: ImplantDesign, uncertainty = 'validated', calibrate = true): Promise<PlanSummary> {
    const r = await check(await fetch(`/api/cases/${id}/plan`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ design, calibrate, uncertainty }),
    }))
    return r.json()
  },
  async planFields(planId: string): Promise<Bundle> {
    return decodeBundle(await (await check(await fetch(`/api/plans/${planId}/fields`))).arrayBuffer())
  },
  async refine(id: string, design: ImplantDesign): Promise<{ state: string }> {
    const r = await check(await fetch(`/api/cases/${id}/refine`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ design }),
    }))
    return r.json()
  },
  async refineStatus(id: string, design: ImplantDesign): Promise<{ state: string; error?: string }> {
    const r = await check(await fetch(`/api/cases/${id}/refine/status`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ design }),
    }))
    return r.json()
  },
  async expressions(id: string): Promise<Bundle> {
    return decodeBundle(await (await check(await fetch(`/api/cases/${id}/expressions`))).arrayBuffer())
  },
  async aging(id: string): Promise<Bundle> {
    return decodeBundle(await (await check(await fetch(`/api/cases/${id}/aging`))).arrayBuffer())
  },
  async getVariants(id: string): Promise<ImplantDesign[]> {
    return (await check(await fetch(`/api/cases/${id}/variants`))).json()
  },
  async putVariants(id: string, v: ImplantDesign[]) {
    await check(await fetch(`/api/cases/${id}/variants`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(v),
    }))
  },
  async updateLandmarks(id: string, lm: Record<string, Vec3>) {
    return (await check(await fetch(`/api/cases/${id}/landmarks`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(lm),
    }))).json()
  },
  async report(id: string, plans: PlanSummary[], images: Record<string, string>): Promise<string> {
    const r = await check(await fetch(`/api/cases/${id}/report`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ plans, images }),
    }))
    return r.text()
  },
}
