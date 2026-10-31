import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Viewer, Layers, ImplantMaterialName, headToThree } from '../viewer/Viewer'
import { FaceSurface, DeformState } from '../viewer/face'
import type { LightingPreset } from '../viewer/lighting'
import type { SkinParams } from '../viewer/skinMaterial'

export type LayoutMode = 'side' | 'wipe' | 'post' | 'pre' | 'compare'

export interface ViewportProps {
  face: FaceSurface | null
  bones: { cranium?: THREE.BufferGeometry; mandible?: THREE.BufferGeometry; teeth?: THREE.BufferGeometry; teethUpper?: THREE.BufferGeometry }
  implant: THREE.BufferGeometry | null
  implantMaterial: ImplantMaterialName
  landmarks: { position: [number, number, number]; kind: 'soft' | 'bone' | 'nerve'; confidence: number }[]
  deform: DeformState
  deformVersion: number
  layout: LayoutMode
  layers: Layers
  lighting: LightingPreset
  lightRotation: number
  skin: SkinParams
  target: [number, number, number]
  postLabel: string
  preLabel: string
  onSnapshot?: (fn: () => Record<string, string>) => void
  viewPreset?: { name: string; n: number }
}

const VIEW_DIRS: Record<string, [number, number, number]> = {
  front: [0, 0, 1],
  oblique_r: [-0.71, 0, 0.71],
  oblique_l: [0.71, 0, 0.71],
  profile_r: [-1, 0, 0],
  profile_l: [1, 0, 0],
  below: [0, -0.7, 0.71],
}

export default function Viewport(p: ViewportProps) {
  const wrap = useRef<HTMLDivElement>(null)
  const preCanvas = useRef<HTMLCanvasElement>(null)
  const postCanvas = useRef<HTMLCanvasElement>(null)
  const overlay = useRef<HTMLDivElement>(null)
  const viewers = useRef<{ pre?: Viewer; post?: Viewer }>({})
  const camera = useMemo(() => new THREE.PerspectiveCamera(20, 1, 5, 3000), [])
  const controls = useRef<OrbitControls>()
  const [wipe, setWipe] = useState(0.5)
  const dirty = useRef(true)
  const target3 = useMemo(() => headToThree(p.target), [p.target[0], p.target[1], p.target[2]])

  // ---- create viewers once
  useEffect(() => {
    if (!preCanvas.current || !postCanvas.current || !overlay.current) return
    const pre = new Viewer(preCanvas.current, camera, target3, p.skin)
    const post = new Viewer(postCanvas.current, camera, target3, p.skin)
    viewers.current = { pre, post }
    camera.position.copy(target3).add(new THREE.Vector3(0, 10, 460))
    const c = new OrbitControls(camera, overlay.current)
    c.target.copy(target3)
    c.enableDamping = true
    c.dampingFactor = 0.12
    c.minDistance = 120
    c.maxDistance = 1400
    c.addEventListener('change', () => { dirty.current = true })
    controls.current = c
    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const moved = c.update()
      if (moved || dirty.current) {
        dirty.current = false
        pre.render()
        post.render()
      }
    }
    loop()
    const ro = new ResizeObserver(() => resize())
    if (wrap.current) ro.observe(wrap.current)
    resize()
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      c.dispose()
      pre.dispose()
      post.dispose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function resize() {
    const el = wrap.current
    if (!el || !viewers.current.pre) return
    const W = el.clientWidth, H = el.clientHeight
    const side = p.layout === 'side'
    const w = side ? Math.floor(W / 2) : W
    viewers.current.pre!.resize(w, H)
    viewers.current.post!.resize(w, H)
    dirty.current = true
  }
  useEffect(resize, [p.layout])

  // ---- data
  useEffect(() => {
    const { pre, post } = viewers.current
    if (!pre || !post || !p.face) return
    pre.setSkinGeometry(p.face.pre)
    post.setSkinGeometry(p.face.post)
    dirty.current = true
  }, [p.face])

  useEffect(() => {
    const { pre, post } = viewers.current
    if (!pre || !post) return
    pre.setBones(p.bones.cranium, p.bones.mandible, p.bones.teeth, p.bones.teethUpper)
    post.setBones(p.bones.cranium, p.bones.mandible, p.bones.teeth, p.bones.teethUpper)
    dirty.current = true
  }, [p.bones])

  useEffect(() => {
    viewers.current.post?.setImplant(p.implant, p.implantMaterial)
    viewers.current.pre?.setImplant(null)
    dirty.current = true
  }, [p.implant, p.implantMaterial])

  useEffect(() => {
    viewers.current.pre?.setLandmarks(p.landmarks)
    viewers.current.post?.setLandmarks(p.landmarks)
    dirty.current = true
  }, [p.landmarks])

  useEffect(() => {
    const { pre, post } = viewers.current
    if (!pre || !post || !p.face) return
    p.face.update(p.deform, p.skin.overlay)
    const jaw = p.face.jawMatrix(p.deform)
    pre.setJaw(jaw)
    post.setJaw(jaw)
    dirty.current = true
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.deformVersion, p.face])

  useEffect(() => {
    viewers.current.pre?.setLayers({ ...p.layers, implant: false })
    viewers.current.post?.setLayers(p.layers)
    dirty.current = true
  }, [p.layers])

  useEffect(() => {
    viewers.current.pre?.setLighting(p.lighting, p.lightRotation)
    viewers.current.post?.setLighting(p.lighting, p.lightRotation)
    dirty.current = true
  }, [p.lighting, p.lightRotation])

  useEffect(() => {
    viewers.current.pre?.setSkin({ ...p.skin, overlay: 0 })
    viewers.current.post?.setSkin(p.skin)
    dirty.current = true
  }, [p.skin])

  useEffect(() => {
    if (!p.viewPreset || !controls.current) return
    const d = VIEW_DIRS[p.viewPreset.name] ?? VIEW_DIRS.front
    const dist = camera.position.distanceTo(controls.current.target)
    camera.position.copy(controls.current.target).add(new THREE.Vector3(...d).normalize().multiplyScalar(dist))
    camera.up.set(0, 1, 0)
    camera.lookAt(controls.current.target)
    dirty.current = true
  }, [p.viewPreset?.n])

  useEffect(() => {
    p.onSnapshot?.(() => {
      const out: Record<string, string> = {}
      if (viewers.current.pre) out.pre = viewers.current.pre.snapshot()
      if (viewers.current.post) out.post = viewers.current.post.snapshot()
      return out
    })
  }, [p.onSnapshot])

  const side = p.layout === 'side'
  const clip = p.layout === 'wipe' ? `inset(0 0 0 ${wipe * 100}%)` : 'none'
  return (
    <div className="viewport" ref={wrap}>
      <canvas ref={preCanvas} className="vp-canvas" style={{
        left: 0, width: side ? '50%' : '100%', display: p.layout === 'post' ? 'none' : 'block',
      }} />
      <canvas ref={postCanvas} className="vp-canvas" style={{
        left: side ? '50%' : 0, width: side ? '50%' : '100%', clipPath: clip,
        display: p.layout === 'pre' ? 'none' : 'block',
      }} />
      <div className="vp-overlay" ref={overlay} />
      {p.layout === 'side' && <div className="vp-divider" style={{ left: '50%' }} />}
      {p.layout === 'wipe' && (
        <div className="vp-wipe" style={{ left: `${wipe * 100}%` }}
             onPointerDown={(e) => {
               const el = wrap.current!
               const move = (ev: PointerEvent) => {
                 const r = el.getBoundingClientRect()
                 setWipe(THREE.MathUtils.clamp((ev.clientX - r.left) / r.width, 0.02, 0.98))
               }
               const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
               window.addEventListener('pointermove', move)
               window.addEventListener('pointerup', up)
               e.stopPropagation()
             }}>
          <div className="vp-wipe-handle">⟷</div>
        </div>
      )}
      <div className="vp-label left">{p.layout === 'post' ? p.postLabel : p.preLabel}</div>
      {(p.layout === 'side' || p.layout === 'wipe') && <div className="vp-label right">{p.postLabel}</div>}
    </div>
  )
}
