import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { api, CaseDetail, CaseMeta, DEFAULT_DESIGN, ImplantDesign, PlanSummary } from './api'
import Viewport, { LayoutMode } from './components/Viewport'
import DesignPanel from './components/DesignPanel'
import { CalibrationBox, LandmarkChanges, MetricsTable, RiskPanel } from './components/AnalysisPanel'
import ProfileChart from './components/ProfileChart'
import { bundleGeometry, DeformState, FaceSurface, planFieldsFromBundle } from './viewer/face'
import VariantGrid, { VariantPanel } from './components/VariantGrid'
import { LIGHTING_LABELS, LightingPreset } from './viewer/lighting'
import { SKIN_TONES, SkinParams } from './viewer/skinMaterial'
import type { Layers } from './viewer/Viewer'

const SOFT = new Set(['soft_glabella', 'soft_nasion', 'pronasale', 'subnasale', 'labrale_superius', 'stomion',
  'labrale_inferius', 'soft_b_point', 'soft_pogonion', 'soft_gnathion', 'soft_menton', 'cervical_point'])

function useDebounced<T>(v: T, ms: number) {
  const [d, setD] = useState(v)
  useEffect(() => { const t = setTimeout(() => setD(v), ms); return () => clearTimeout(t) }, [v, ms])
  return d
}

export default function App() {
  const [cases, setCases] = useState<CaseMeta[]>([])
  const [caseId, setCaseId] = useState<string | null>(null)
  const [detail, setDetail] = useState<CaseDetail | null>(null)
  const [face, setFace] = useState<FaceSurface | null>(null)
  const [bones, setBones] = useState<{ cranium?: THREE.BufferGeometry; mandible?: THREE.BufferGeometry; teeth?: THREE.BufferGeometry; teethUpper?: THREE.BufferGeometry }>({})
  const [variants, setVariants] = useState<ImplantDesign[]>([{ ...DEFAULT_DESIGN }])
  const [active, setActive] = useState(0)
  const design = variants[active]
  const [implantGeo, setImplantGeo] = useState<THREE.BufferGeometry | null>(null)
  const [plans, setPlans] = useState<Record<number, PlanSummary>>({})
  const plan = plans[active] ?? null
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [refineState, setRefineState] = useState('none')
  const [layout, setLayout] = useState<LayoutMode>('side')
  const [timeline, setTimeline] = useState(6)
  const [range, setRange] = useState(0)
  const [expression, setExpression] = useState('neutral')
  const [exprAmount, setExprAmount] = useState(1)
  const [agingYears, setAgingYears] = useState(0)
  const [layers, setLayers] = useState<Layers>({ skinOpacity: 1, bone: false, implant: true, landmarks: false })
  const [lighting, setLighting] = useState<LightingPreset>('studio')
  const [lightRot, setLightRot] = useState(0)
  const [tone, setTone] = useState('III – medium')
  const [beard, setBeard] = useState(0)
  const [overlay, setOverlay] = useState<0 | 1 | 2>(0)
  const [uncProfile, setUncProfile] = useState<'validated' | 'relaxed'>('validated')
  const [rightTab, setRightTab] = useState<'analysis' | 'risks' | 'profile' | 'display'>('analysis')
  const [viewPreset, setViewPreset] = useState({ name: 'front', n: 0 })
  const [deformVersion, setDeformVersion] = useState(0)
  const [exprList, setExprList] = useState<Record<string, { label: string }>>({})
  const snapshotFn = useRef<() => Record<string, string>>()
  const [compareCache, setCompareCache] = useState<Record<string, Omit<VariantPanel, 'design' | 'loading'>>>({})
  const [compareBusy, setCompareBusy] = useState<string | null>(null)

  // ---------------- cases
  const refreshCases = useCallback(async () => {
    try {
      const c = await api.listCases()
      setCases(c)
      if (!caseId && c.length) setCaseId((c.find((x) => x.status.stage === 'ready' && x.anatomy_source !== 'estimated')
        ?? c.find((x) => x.status.stage === 'ready') ?? c[0]).id)
    } catch (e: any) { setError(e.message) }
  }, [caseId])
  useEffect(() => { refreshCases() }, [refreshCases])

  useEffect(() => {
    if (!caseId) return
    let stop = false
    setDetail(null); setFace(null); setPlans({}); setBones({}); setImplantGeo(null)
    const load = async () => {
      const d = await api.getCase(caseId)
      if (stop) return
      setDetail(d)
      if (d.meta.status.stage !== 'ready') {
        setTimeout(() => !stop && load(), 3000)
        return
      }
      const opt = (n: string) => (d.meshes?.includes(n) ? api.mesh(caseId, n) : Promise.resolve(null))
      const [skin, mand, cran, teeth, teethU, vs] = await Promise.all([
        api.mesh(caseId, 'skin'), api.mesh(caseId, 'mandible'), api.mesh(caseId, 'cranium'),
        opt('teeth_lower'), opt('teeth_upper'), api.getVariants(caseId)])
      if (stop) return
      const fis = typeof d.fissure === 'object' && d.fissure ? d.fissure : null
      const f = new FaceSurface(skin, d.anatomy!.landmarks, fis?.half_width_mm ?? 22, fis?.z)
      setFace(f)
      setBones({ mandible: bundleGeometry(mand), cranium: bundleGeometry(cran), teeth: teeth ? bundleGeometry(teeth) : undefined,
        teethUpper: teethU ? bundleGeometry(teethU) : undefined })
      if (vs.length) { setVariants(vs.map((v) => ({ ...DEFAULT_DESIGN, ...v }))); setActive(0) }
      api.expressions(caseId).then((b) => { if (!stop) { f.setExpressions(b); setExprList(b.meta.expressions); setDeformVersion((v) => v + 1) } })
        .catch((e) => console.warn('expressions', e))
      api.aging(caseId).then((b) => { if (!stop) { f.setAging(b); setDeformVersion((v) => v + 1) } })
        .catch((e) => console.warn('aging', e))
    }
    load().catch((e) => setError(e.message))
    return () => { stop = true }
  }, [caseId])

  // ---------------- prediction (debounced)
  const dDesign = useDebounced(design, 450)
  const reqId = useRef(0)
  useEffect(() => {
    if (!caseId || !face) return
    const id = ++reqId.current
    setBusy(true)
    setError(null)
    const run = async () => {
      const [imp, pl] = await Promise.all([api.implant(caseId, dDesign), api.plan(caseId, dDesign, uncProfile)])
      if (id !== reqId.current) return
      setImplantGeo(bundleGeometry(imp))
      const fields = await api.planFields(pl.plan_id)
      if (id !== reqId.current) return
      face.setPlan(fields)
      setPlans((p) => ({ ...p, [active]: pl }))
      setRefineState(pl.nonlinear ? 'done' : 'none')
      setDeformVersion((v) => v + 1)
    }
    run().catch((e) => setError(e.message)).finally(() => { if (id === reqId.current) setBusy(false) })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dDesign, caseId, face, uncProfile])

  // ---------------- variant comparison: compute every variant's prediction, one at a time
  const vkey = (d: ImplantDesign) => JSON.stringify({ ...d, name: '' }) + uncProfile
  useEffect(() => {
    if (layout !== 'compare' || !caseId || !face) return
    let stop = false
    const run = async () => {
      for (const v of variants.slice(0, 3)) {
        const k = vkey(v)
        if (stop || compareCache[k]) continue
        setCompareBusy(k)
        try {
          const [imp, pl] = await Promise.all([api.implant(caseId, v), api.plan(caseId, v, uncProfile)])
          const fields = planFieldsFromBundle(await api.planFields(pl.plan_id))
          if (stop) return
          setCompareCache((c) => ({ ...c, [k]: { plan: pl, fields, implant: bundleGeometry(imp) } }))
        } catch (e: any) { setError(e.message) }
      }
      if (!stop) setCompareBusy(null)
    }
    run()
    return () => { stop = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, variants, caseId, face, uncProfile, compareCache])
  useEffect(() => { setCompareCache({}) }, [caseId])
  const panels: VariantPanel[] = useMemo(() => variants.slice(0, 3).map((v) => {
    const c = compareCache[vkey(v)]
    return { design: v, plan: c?.plan ?? null, fields: c?.fields, implant: c?.implant ?? null, loading: compareBusy === vkey(v) || !c }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [variants, compareCache, compareBusy, uncProfile])

  // save variants
  const dVariants = useDebounced(variants, 1500)
  useEffect(() => { if (caseId && face) api.putVariants(caseId, dVariants).catch(() => undefined) }, [dVariants, caseId, face])

  // ---------------- deformation state
  const deform: DeformState = useMemo(() => ({
    timeline, range, agingYears,
    expression: expression === 'neutral' ? {} : { [expression]: exprAmount },
  }), [timeline, range, agingYears, expression, exprAmount])
  useEffect(() => setDeformVersion((v) => v + 1), [deform])

  const skin: SkinParams = useMemo(() => {
    const t = new THREE.Color(SKIN_TONES[tone])
    const lip = t.clone().multiply(new THREE.Color('#c97e7e')).multiplyScalar(1.05)
    return { tone: t, lip, beard, sss: 1, detail: 1, oil: 0.4, overlay, overlayScale: overlay === 1 ? 3 : 6, hasTexture: false }
  }, [tone, beard, overlay])

  const landmarks = useMemo(() => {
    if (!detail?.anatomy) return []
    return Object.values(detail.anatomy.landmarks).map((l) => ({
      position: l.position, confidence: l.confidence,
      kind: (l.name.startsWith('mental_foramen') ? 'nerve' : SOFT.has(l.name) ? 'soft' : 'bone') as 'soft' | 'bone' | 'nerve',
    }))
  }, [detail])

  const target = useMemo<[number, number, number]>(() => {
    const L = detail?.anatomy?.landmarks
    if (!L?.soft_pogonion || !L.subnasale) return [0, 40, -30]
    const a = L.soft_pogonion.position, b = L.subnasale.position
    return [0, (a[1] + b[1]) / 2 - 25, (a[2] + b[2]) / 2 + 8]
  }, [detail])

  const tps = plan?.timepoints ?? []
  const tLabel = tps.length ? tps[Math.round(timeline)]?.label ?? '' : ''
  const postLabel = `Predicted · ${design.name}${tLabel ? ' · ' + tLabel : ''}${range !== 0 ? (range < 0 ? ' · conservative' : ' · generous') : ''}`
  const preLabel = `Current${agingYears ? ` · +${agingYears} yrs` : ''}`

  // ---------------- actions
  const updateDesign = (d: ImplantDesign) => setVariants((vs) => vs.map((v, i) => (i === active ? d : v)))
  const addVariant = () => {
    const name = `Variant ${String.fromCharCode(65 + variants.length)}`
    setVariants((vs) => [...vs, { ...design, name }])
    setActive(variants.length)
  }
  const deleteVariant = (i: number) => {
    setVariants((vs) => vs.filter((_, j) => j !== i))
    setPlans({})
    setActive(0)
  }
  const exportStl = async () => {
    if (!caseId) return
    const blob = await api.implantStl(caseId, design)
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${design.name.replace(/\s+/g, '_')}.stl`
    a.click()
  }
  const refine = async () => {
    if (!caseId) return
    const st = await api.refine(caseId, design)
    setRefineState(st.state)
    const poll = setInterval(async () => {
      const s = await api.refineStatus(caseId, design)
      setRefineState(s.state)
      if (s.state === 'done' || s.state === 'failed') {
        clearInterval(poll)
        if (s.state === 'done') setVariants((vs) => [...vs])   // re-run prediction with the correction
        if (s.state === 'failed') setError('Nonlinear refinement failed: ' + s.error)
      }
    }, 4000)
  }
  const report = async () => {
    if (!caseId) return
    const imgs = snapshotFn.current?.() ?? {}
    const ps = Object.values(plans)
    const html = await api.report(caseId, ps, imgs)
    const w = window.open('', '_blank')
    if (w) { w.document.write(html); w.document.close() }
  }
  const upload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || !files.length) return
    const name = prompt('Case name', files[0].name.replace(/\.[^.]+$/, '')) || 'New case'
    const sex = prompt('Sex (F/M, optional)', '') || ''
    const fd = new FormData()
    for (const f of Array.from(files)) fd.append('files', f)
    fd.append('name', name)
    fd.append('sex', sex)
    try {
      const m = await api.createCase(fd)
      await refreshCases()
      setCaseId(m.id)
    } catch (err: any) { setError(err.message) }
    e.target.value = ''
  }

  const status = detail?.meta.status
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Mento<span>Plan</span></div>
        <select className="case-select" value={caseId ?? ''} onChange={(e) => setCaseId(e.target.value)}>
          {cases.map((c) => <option key={c.id} value={c.id}>{c.name} — {c.status.stage}</option>)}
        </select>
        <label className="btn ghost small upload">
          Import CT / CBCT / face scan
          <input type="file" multiple accept=".dcm,.zip,.nrrd,.nhdr,.nii,.gz,.mha,.mhd,.gipl,.obj,.mtl,.jpg,.jpeg,.png,.ply,.stl" onChange={upload} hidden />
        </label>
        <div className="spacer" />
        <div className="layout-switch">
          {(['side', 'wipe', 'post', 'pre', 'compare'] as LayoutMode[]).map((m) => (
            <button key={m} className={layout === m ? 'on' : ''} onClick={() => setLayout(m)}
                    title={m === 'compare' ? 'Current face and up to three implant variants side by side' : undefined}>
              {{ side: 'Side by side', wipe: 'Wipe', post: 'Predicted', pre: 'Current', compare: 'Compare variants' }[m]}
            </button>
          ))}
        </div>
        <div className="view-switch">
          {[['front', 'Front'], ['oblique_r', '¾ R'], ['profile_r', 'Profile R'], ['oblique_l', '¾ L'], ['profile_l', 'Profile L'], ['below', 'Below']].map(([k, l]) => (
            <button key={k} onClick={() => setViewPreset({ name: k, n: viewPreset.n + 1 })}>{l}</button>
          ))}
        </div>
        <button className="btn small" onClick={report} disabled={!plan}>Report</button>
      </header>

      <aside className="left">
        {status && status.stage !== 'ready' && (
          <div className="processing">
            <div>Processing scan: {status.message}</div>
            <div className="bar"><div style={{ width: `${status.progress * 100}%` }} /></div>
          </div>
        )}
        {detail?.meta.anatomy_source === 'estimated' && (
          <div className="warning" style={{ margin: 12 }}>
            Planning from a face scan only: bone, nerves and tissue layers are estimated from a template head and
            population norms. Ranges are wider; obtain a CT/CBCT before surgery.
          </div>
        )}
        <DesignPanel design={design} onChange={updateDesign} variants={variants} active={active}
                     onSelectVariant={(i) => setActive(i)} onAddVariant={addVariant} onDeleteVariant={deleteVariant}
                     onExportStl={exportStl} onRefine={refine} refineState={refineState} busy={busy || !face} />
      </aside>

      <main className="center">
        {layout === 'compare' && face ? (
          <VariantGrid face={face} bones={bones} panels={panels} deform={deform} deformVersion={deformVersion}
                       lighting={lighting} lightRotation={lightRot} skin={skin} target={target}
                       showImplant={layers.implant && layers.skinOpacity < 0.999} tIndex={timeline}
                       viewPreset={viewPreset} />
        ) : (
        <Viewport face={face} bones={bones} implant={implantGeo} implantMaterial={design.material} landmarks={landmarks}
                  deform={deform} deformVersion={deformVersion} layout={layout} layers={layers} lighting={lighting}
                  lightRotation={lightRot} skin={skin} target={target} postLabel={postLabel} preLabel={preLabel}
                  viewPreset={viewPreset} onSnapshot={(fn) => { snapshotFn.current = fn }} />
        )}
        {busy && <div className="busy">computing prediction…</div>}
        {error && <div className="error" onClick={() => setError(null)}>{error}</div>}
        <div className="timeline">
          <div className="tl-row">
            <span className="tl-title">Healing</span>
            <input type="range" min={0} max={Math.max(tps.length - 1, 0)} step={0.01} value={Math.min(timeline, Math.max(tps.length - 1, 0))}
                   onChange={(e) => setTimeline(parseFloat(e.target.value))} disabled={!tps.length} />
            <div className="tl-ticks">{tps.map((t, i) => <span key={i} onClick={() => setTimeline(i)}
                 className={Math.round(timeline) === i ? 'on' : ''}>{t.label.replace(' (settled)', '')}</span>)}</div>
          </div>
          <div className="tl-row">
            <span className="tl-title" title="Explore the 90% prediction interval along the skin surface">Uncertainty</span>
            <input type="range" min={-1} max={1} step={0.01} value={range} onChange={(e) => setRange(parseFloat(e.target.value))} />
            <div className="tl-ticks range"><span onClick={() => setRange(-1)}>5th pct (less)</span>
              <select className="unc-profile" value={uncProfile} onChange={(e) => setUncProfile(e.target.value as 'validated' | 'relaxed')}
                      title="Validated: calibrated on a real surgical outcome incl. a change of lip/mentalis posture. Relaxed: assumes the patient is relaxed with lips together before and after.">
                <option value="validated">ranges: validated (conservative)</option>
                <option value="relaxed">ranges: relaxed posture</option>
              </select><span onClick={() => setRange(0)}>likely</span><span onClick={() => setRange(1)}>95th pct (more)</span></div>
          </div>
          <div className="tl-row compact">
            <span className="tl-title">Expression</span>
            <select value={expression} onChange={(e) => setExpression(e.target.value)}>
              <option value="neutral">Neutral</option>
              {Object.entries(exprList).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
            <input type="range" min={0} max={1} step={0.01} value={exprAmount} onChange={(e) => setExprAmount(parseFloat(e.target.value))}
                   disabled={expression === 'neutral'} />
            <span className="tl-title">Aging</span>
            <input type="range" min={0} max={20} step={1} value={agingYears} onChange={(e) => setAgingYears(parseFloat(e.target.value))} />
            <span className="mono">+{agingYears} y</span>
          </div>
        </div>
      </main>

      <aside className="right">
        <nav className="tabs">
          {(['analysis', 'risks', 'profile', 'display'] as const).map((t) => (
            <button key={t} className={rightTab === t ? 'on' : ''} onClick={() => setRightTab(t)}>
              {{ analysis: 'Analysis', risks: 'Safety', profile: 'Profile', display: 'Display' }[t]}
            </button>
          ))}
        </nav>
        {rightTab === 'analysis' && (
          <div className="panel">
            {plan && plan.warnings.length > 0 && plan.warnings.map((w, i) => <div className="warning" key={i}>⚠ {w}</div>)}
            <h3>Soft-tissue change {tLabel && <small>· {tLabel}</small>}</h3>
            <LandmarkChanges plan={plan} tIndex={timeline} />
            <CalibrationBox plan={plan} />
            <h3>Facial balance</h3>
            {detail?.preop_metrics && <MetricsTable pre={detail.preop_metrics} plan={plan} tIndex={timeline} />}
            <div className="foot">† depends on natural head position. Ranges are 90% prediction intervals.</div>
            {detail?.anatomy && <div className="notes">{detail.anatomy.notes.map((n, i) => <div key={i}>{n}</div>)}</div>}
          </div>
        )}
        {rightTab === 'risks' && <div className="panel"><RiskPanel plan={plan} /></div>}
        {rightTab === 'profile' && (
          <div className="panel">
            <ProfileChart profile={detail?.anatomy?.profiles.soft_midline} bone={detail?.anatomy?.profiles.bone_symphysis}
                          landmarks={detail?.anatomy?.landmarks ?? {}} plan={plan} tIndex={timeline} />
          </div>
        )}
        {rightTab === 'display' && (
          <div className="panel display">
            <label className="field"><span>Lighting</span>
              <select value={lighting} onChange={(e) => setLighting(e.target.value as LightingPreset)}>
                {Object.entries(LIGHTING_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </label>
            <div className="slider"><div className="slider-head"><span>Light direction</span><span className="mono">{Math.round(lightRot * 57.3)}°</span></div>
              <input type="range" min={-3.14} max={3.14} step={0.01} value={lightRot} onChange={(e) => setLightRot(parseFloat(e.target.value))} /></div>
            <label className="field"><span>Skin tone</span>
              <select value={tone} onChange={(e) => setTone(e.target.value)}>
                {Object.keys(SKIN_TONES).map((k) => <option key={k}>{k}</option>)}
              </select>
            </label>
            <div className="slider"><div className="slider-head"><span>Beard shadow</span><span className="mono">{beard.toFixed(2)}</span></div>
              <input type="range" min={0} max={1} step={0.05} value={beard} onChange={(e) => setBeard(parseFloat(e.target.value))} /></div>
            <label className="field"><span>Surface overlay</span>
              <select value={overlay} onChange={(e) => setOverlay(parseInt(e.target.value) as 0 | 1 | 2)}>
                <option value={0}>None (photoreal)</option>
                <option value={1}>Uncertainty (90% band width, 0–3 mm)</option>
                <option value={2}>Change map (± 6 mm)</option>
              </select>
            </label>
            <div className="slider"><div className="slider-head"><span>Skin opacity (X-ray)</span><span className="mono">{layers.skinOpacity.toFixed(2)}</span></div>
              <input type="range" min={0} max={1} step={0.05} value={layers.skinOpacity}
                     onChange={(e) => setLayers({ ...layers, skinOpacity: parseFloat(e.target.value) })} /></div>
            {(['bone', 'implant', 'landmarks'] as const).map((k) => (
              <label className="check" key={k}>
                <input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} />
                {{ bone: 'Bone & teeth', implant: 'Implant', landmarks: 'Landmarks (red = mental foramen)' }[k]}
              </label>
            ))}
          </div>
        )}
      </aside>
    </div>
  )
}
