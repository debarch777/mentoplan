import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ImplantDesign, PlanSummary } from '../api'
import { Viewer, headToThree } from '../viewer/Viewer'
import { DeformState, FaceSurface, PlanFields } from '../viewer/face'
import type { LightingPreset } from '../viewer/lighting'
import type { SkinParams } from '../viewer/skinMaterial'

export interface VariantPanel {
  design: ImplantDesign
  plan: PlanSummary | null
  fields: PlanFields | undefined
  implant: THREE.BufferGeometry | null
  loading: boolean
}

interface Props {
  face: FaceSurface
  bones: { cranium?: THREE.BufferGeometry; mandible?: THREE.BufferGeometry; teeth?: THREE.BufferGeometry; teethUpper?: THREE.BufferGeometry }
  panels: VariantPanel[]
  deform: DeformState
  deformVersion: number
  lighting: LightingPreset
  lightRotation: number
  skin: SkinParams
  target: [number, number, number]
  showImplant: boolean
  tIndex: number
  viewPreset?: { name: string; n: number }
}

const DIRS: Record<string, [number, number, number]> = {
  front: [0, 0, 1], oblique_r: [-0.71, 0, 0.71], oblique_l: [0.71, 0, 0.71], profile_r: [-1, 0, 0],
  profile_l: [1, 0, 0], below: [0, -0.7, 0.71],
}

function summary(p: VariantPanel, tIndex: number) {
  const tp = p.plan?.timepoints[Math.round(tIndex)]
  const L = tp?.landmarks.soft_pogonion
  if (!L) return p.loading ? 'computing…' : ''
  const d = (k: 'median' | 'p05' | 'p95') => (L[k][1] - L.pre[1]).toFixed(1)
  return `chin +${d('median')} mm [${d('p05')}, ${d('p95')}] · risk ${p.plan!.overall_risk}`
}

/** Current face plus up to three implant variants, side by side with one shared camera. */
export default function VariantGrid(p: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const overlay = useRef<HTMLDivElement>(null)
  const canvases = useRef<(HTMLCanvasElement | null)[]>([])
  const viewers = useRef<Viewer[]>([])
  const geoms = useRef<THREE.BufferGeometry[]>([])
  const camera = useMemo(() => new THREE.PerspectiveCamera(20, 1, 5, 3000), [])
  const dirty = useRef(true)
  const ctlRef = useRef<OrbitControls>()
  const n = 1 + p.panels.length
  const target3 = useMemo(() => headToThree(p.target), [p.target[0], p.target[1], p.target[2]])

  useEffect(() => {
    const vs: Viewer[] = []
    for (let i = 0; i < n; i++) {
      const c = canvases.current[i]
      if (!c) continue
      const v = new Viewer(c, camera, target3, p.skin)
      v.setSkinGeometry(i === 0 ? p.face.pre : (geoms.current[i - 1] ??= p.face.variantGeometry()))
      v.setBones(p.bones.cranium, p.bones.mandible, p.bones.teeth, p.bones.teethUpper)
      vs.push(v)
    }
    viewers.current = vs
    camera.position.copy(target3).add(new THREE.Vector3(-260, 10, 380))
    const ctl = new OrbitControls(camera, overlay.current!)
    ctl.target.copy(target3)
    ctl.enableDamping = true
    ctl.addEventListener('change', () => { dirty.current = true })
    ctlRef.current = ctl
    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      if (ctl.update() || dirty.current) {
        dirty.current = false
        for (const v of vs) v.render()
      }
    }
    loop()
    const resize = () => {
      const el = wrap.current
      if (!el) return
      const cols = n <= 2 ? n : 2
      const rows = Math.ceil(n / cols)
      for (const v of vs) v.resize(Math.floor(el.clientWidth / cols), Math.floor(el.clientHeight / rows))
      dirty.current = true
    }
    const ro = new ResizeObserver(resize)
    ro.observe(wrap.current!)
    resize()
    return () => { cancelAnimationFrame(raf); ro.disconnect(); ctl.dispose(); vs.forEach((v) => v.dispose()) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n, p.face])

  useEffect(() => {
    // the single-view Viewport is unmounted in this layout: compose the shared current face
    // (expression + aging) here before deriving each variant from it
    p.face.update(p.deform, 0)
    p.panels.forEach((panel, i) => {
      const g = geoms.current[i]
      if (g) p.face.updateVariant(g, panel.fields, p.deform)
      const v = viewers.current[i + 1]
      v?.setImplant(panel.implant, panel.design.material)
      v?.setLayers({ implant: p.showImplant, skinOpacity: 1 })
    })
    const jaw = p.face.jawMatrix(p.deform)
    viewers.current.forEach((v) => v.setJaw(jaw))
    dirty.current = true
  }, [p.panels, p.deformVersion, p.showImplant, p.face, n])

  useEffect(() => {
    viewers.current.forEach((v) => { v.setLighting(p.lighting, p.lightRotation); v.setSkin(p.skin) })
    dirty.current = true
  }, [p.lighting, p.lightRotation, p.skin, n])

  useEffect(() => {
    const ctl = ctlRef.current
    if (!p.viewPreset || !ctl) return
    const d = DIRS[p.viewPreset.name] ?? DIRS.front
    const dist = camera.position.distanceTo(ctl.target)
    camera.position.copy(ctl.target).add(new THREE.Vector3(...d).normalize().multiplyScalar(dist))
    camera.lookAt(ctl.target)
    dirty.current = true
  }, [p.viewPreset?.n])

  const cols = n <= 2 ? n : 2
  return (
    <div className="viewport variant-grid" ref={wrap}
         style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {Array.from({ length: n }).map((_, i) => (
        <div className="vg-cell" key={i}>
          <canvas ref={(el) => { canvases.current[i] = el }} className="vg-canvas" />
          <div className="vp-label left">{i === 0 ? 'Current' : p.panels[i - 1].design.name}</div>
          {i > 0 && <div className="vg-sub">{summary(p.panels[i - 1], p.tIndex)}</div>}
        </div>
      ))}
      <div className="vp-overlay" ref={overlay} />
    </div>
  )
}
