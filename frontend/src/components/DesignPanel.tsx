import type { ImplantDesign } from '../api'

interface SliderDef {
  key: keyof ImplantDesign
  label: string
  min: number
  max: number
  step: number
  unit: string
  help: string
  show?: (d: ImplantDesign) => boolean
}

const GROUPS: { title: string; sliders: SliderDef[] }[] = [
  {
    title: 'Size',
    sliders: [
      { key: 'projection', label: 'Projection', min: 0, max: 14, step: 0.5, unit: 'mm', help: 'Maximum anterior thickness (sagittal augmentation)' },
      { key: 'width', label: 'Width (arc)', min: 20, max: 80, step: 1, unit: 'mm', help: 'Footprint length along the mandibular arch, wings included' },
      { key: 'height', label: 'Height', min: 8, max: 30, step: 0.5, unit: 'mm', help: 'Vertical extent on the labial face of the symphysis' },
      { key: 'inferior_wrap', label: 'Under-border wrap', min: 0, max: 4, step: 0.25, unit: 'mm', help: 'Thin lip of implant wrapping under the inferior border (stability)' },
      { key: 'vertical_extension', label: 'Vertical lengthening', min: 0, max: 8, step: 0.5, unit: 'mm', help: 'Thickness wrapped under the inferior border – lengthens the chin' },
    ],
  },
  {
    title: 'Position',
    sliders: [
      { key: 'inferior_offset', label: 'Height above border', min: 0, max: 10, step: 0.5, unit: 'mm', help: 'Lower edge relative to the inferior border (0 = seated on the border)' },
      { key: 'midline_shift', label: 'Midline shift', min: -6, max: 6, step: 0.25, unit: 'mm', help: '+ moves the implant towards the patient’s right' },
      { key: 'cant_deg', label: 'Cant (roll)', min: -10, max: 10, step: 0.5, unit: '°', help: 'Coronal rotation; + raises the patient’s right side' },
    ],
  },
  {
    title: 'Shape',
    sliders: [
      { key: 'peak_height', label: 'Peak position', min: 0.15, max: 0.85, step: 0.01, unit: '', help: 'Where the maximum projection sits (0 = at the border, 1 = top edge)' },
      { key: 'base_fill', label: 'Border fullness', min: 0, max: 1, step: 0.05, unit: '', help: 'Thickness at the inferior border relative to the peak' },
      { key: 'squareness', label: 'Squareness', min: 0, max: 1, step: 0.05, unit: '', help: 'Pointed/round (0) to broad and square (1) frontal outline' },
      { key: 'wing_ratio', label: 'Wing thickness', min: 0, max: 0.8, step: 0.05, unit: '×', help: 'Lateral wing thickness relative to the projection', show: (d) => d.shape === 'anatomical' || d.shape === 'custom' },
      { key: 'body_fraction', label: 'Central body width', min: 0.25, max: 0.9, step: 0.05, unit: '×', help: 'Width of the central body relative to the full width', show: (d) => d.shape === 'anatomical' || d.shape === 'custom' },
      { key: 'asymmetry', label: 'Left / right asymmetry', min: -0.6, max: 0.6, step: 0.05, unit: '', help: 'Extra projection on the patient’s right (+) or left (–) to correct asymmetry' },
      { key: 'edge_feather', label: 'Edge feathering', min: 2, max: 12, step: 0.5, unit: 'mm', help: 'Taper length at the edges (longer = less palpable edge)' },
    ],
  },
]

export const SHAPE_PRESETS: Record<string, Partial<ImplantDesign>> = {
  'Central (round)': { shape: 'round', width: 34, height: 17, projection: 6, squareness: 0.1 },
  'Extended anatomical': { shape: 'anatomical', width: 56, height: 18, projection: 6, wing_ratio: 0.35, body_fraction: 0.5, squareness: 0.3 },
  'Square (masculine)': { shape: 'square', width: 52, height: 19, projection: 6, squareness: 0.75 },
  'Narrow (feminine)': { shape: 'round', width: 30, height: 15, projection: 5, squareness: 0.0, peak_height: 0.5 },
  'Vertical lengthening': { shape: 'anatomical', width: 50, height: 18, projection: 4, vertical_extension: 4, base_fill: 0.9 },
}

interface Props {
  design: ImplantDesign
  onChange: (d: ImplantDesign) => void
  variants: ImplantDesign[]
  active: number
  onSelectVariant: (i: number) => void
  onAddVariant: () => void
  onDeleteVariant: (i: number) => void
  onExportStl: () => void
  onRefine: () => void
  refineState: string
  busy: boolean
}

export default function DesignPanel(p: Props) {
  const d = p.design
  const set = (k: keyof ImplantDesign, v: any) => p.onChange({ ...d, [k]: v })
  return (
    <div className="panel design-panel">
      <div className="variant-tabs">
        {p.variants.map((v, i) => (
          <button key={i} className={'variant-tab' + (i === p.active ? ' active' : '')} onClick={() => p.onSelectVariant(i)}>
            {v.name}
            {p.variants.length > 1 && i === p.active && (
              <span className="x" title="Delete variant" onClick={(e) => { e.stopPropagation(); p.onDeleteVariant(i) }}>×</span>
            )}
          </button>
        ))}
        <button className="variant-tab add" onClick={p.onAddVariant} title="Duplicate as new variant">+</button>
      </div>
      <div className="row">
        <label className="field grow">
          <span>Name</span>
          <input value={d.name} onChange={(e) => set('name', e.target.value)} />
        </label>
      </div>
      <div className="presets">
        {Object.entries(SHAPE_PRESETS).map(([k, v]) => (
          <button key={k} className="chip" onClick={() => p.onChange({ ...d, ...v })}>{k}</button>
        ))}
      </div>
      <div className="row">
        <label className="field">
          <span>Outline</span>
          <select value={d.shape} onChange={(e) => set('shape', e.target.value)}>
            <option value="round">Central / round</option>
            <option value="anatomical">Anatomical (winged)</option>
            <option value="square">Square</option>
          </select>
        </label>
        <label className="field">
          <span>Material</span>
          <select value={d.material} onChange={(e) => set('material', e.target.value)}>
            <option value="silicone">Silicone</option>
            <option value="porous_polyethylene">Porous polyethylene</option>
            <option value="peek">PEEK (custom milled)</option>
            <option value="titanium">Titanium (printed)</option>
          </select>
        </label>
        <label className="field">
          <span>Fixation</span>
          <select value={d.fixation} onChange={(e) => set('fixation', e.target.value)}>
            <option value="screws">Screws</option>
            <option value="sutures">Sutures</option>
            <option value="none">None (pocket)</option>
          </select>
        </label>
      </div>
      {GROUPS.map((g) => (
        <fieldset key={g.title} className="group">
          <legend>{g.title}</legend>
          {g.sliders.filter((s) => !s.show || s.show(d)).map((s) => (
            <div className="slider" key={s.key} title={s.help}>
              <div className="slider-head">
                <span>{s.label}</span>
                <span className="mono">{(d[s.key] as number).toFixed(s.step < 0.1 ? 2 : 1)} {s.unit}</span>
              </div>
              <input type="range" min={s.min} max={s.max} step={s.step} value={d[s.key] as number}
                     onChange={(e) => set(s.key, parseFloat(e.target.value))} />
            </div>
          ))}
        </fieldset>
      ))}
      <div className="row actions">
        <button className="btn" onClick={p.onExportStl} disabled={p.busy}>Export implant STL</button>
        <button className="btn ghost" onClick={p.onRefine} disabled={p.busy || p.refineState === 'running' || p.refineState === 'queued'}
                title="Large-deformation (neo-Hookean) solution of the current design; takes a few minutes">
          {p.refineState === 'running' || p.refineState === 'queued' ? 'Refining…' : p.refineState === 'done' ? 'Nonlinear ✓' : 'Nonlinear refine'}
        </button>
      </div>
    </div>
  )
}
