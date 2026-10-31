import * as THREE from 'three'

// Procedural HDR lighting environments.  Each preset is a small emissive "studio" scene that is
// prefiltered with PMREM for image-based lighting (diffuse + glossy reflections), plus a matching
// shadow-casting key light so that soft shadows under the chin and in the labiomental fold appear.

export type LightingPreset = 'studio' | 'clinical' | 'window' | 'overhead' | 'evening'

export const LIGHTING_LABELS: Record<LightingPreset, string> = {
  studio: 'Studio softbox',
  clinical: 'Clinical photo (flash)',
  window: 'Window daylight',
  overhead: 'Overhead (harsh)',
  evening: 'Warm evening',
}

interface Box { pos: [number, number, number]; size: [number, number]; color: string; intensity: number }
interface PresetDef {
  bg: [string, string]          // gradient top/bottom of the room
  boxes: Box[]
  key: { dir: [number, number, number]; color: string; intensity: number; soft: number }
  fill: { color: string; intensity: number }
  exposure: number
}

// directions are in the viewer frame: +x patient left (viewer right), +y up, +z towards the face front
const PRESETS: Record<LightingPreset, PresetDef> = {
  studio: {
    bg: ['#2b2a2d', '#141416'],
    boxes: [
      { pos: [-2.2, 1.6, 2.4], size: [2.4, 2.4], color: '#fff6ec', intensity: 9 },
      { pos: [2.6, 0.6, 1.8], size: [1.6, 2.6], color: '#eef3ff', intensity: 3.2 },
      { pos: [0.0, 1.0, -3.0], size: [3.0, 1.2], color: '#ffffff', intensity: 2.5 },
      { pos: [0, -2.5, 1.5], size: [3, 1.5], color: '#a89a90', intensity: 0.8 },
    ],
    key: { dir: [-0.55, 0.5, 0.67], color: '#fff3e6', intensity: 2.2, soft: 4 },
    fill: { color: '#dfe6f2', intensity: 0.15 },
    exposure: 1.0,
  },
  clinical: {
    bg: ['#d9dcdf', '#bfc3c7'],
    boxes: [
      { pos: [0, 0.35, 3.2], size: [1.2, 1.2], color: '#ffffff', intensity: 14 },
      { pos: [-2.5, 0.5, 1.2], size: [1.5, 2.5], color: '#ffffff', intensity: 2 },
      { pos: [2.5, 0.5, 1.2], size: [1.5, 2.5], color: '#ffffff', intensity: 2 },
    ],
    key: { dir: [0.02, 0.12, 1.0], color: '#ffffff', intensity: 1.7, soft: 2 },
    fill: { color: '#ffffff', intensity: 0.25 },
    exposure: 0.9,
  },
  window: {
    bg: ['#9fb6cf', '#6a6258'],
    boxes: [
      { pos: [3.2, 1.2, 1.2], size: [2.2, 3.2], color: '#e9f1ff', intensity: 8 },
      { pos: [0, 3.5, 0], size: [5, 5], color: '#c8d7ea', intensity: 1.2 },
      { pos: [-3, 0, 0.5], size: [2, 3], color: '#d9c6ad', intensity: 0.7 },
    ],
    key: { dir: [0.85, 0.35, 0.4], color: '#f4f7ff', intensity: 2.0, soft: 6 },
    fill: { color: '#d8c8b0', intensity: 0.2 },
    exposure: 1.0,
  },
  overhead: {
    bg: ['#3a3a3a', '#0e0e0e'],
    boxes: [
      { pos: [0, 3.2, 0.6], size: [1.6, 1.6], color: '#ffffff', intensity: 16 },
      { pos: [0, -3, 1], size: [3, 2], color: '#555555', intensity: 0.3 },
    ],
    key: { dir: [0.0, 0.94, 0.33], color: '#ffffff', intensity: 2.6, soft: 2 },
    fill: { color: '#ffffff', intensity: 0.05 },
    exposure: 1.0,
  },
  evening: {
    bg: ['#3a2a22', '#120c0a'],
    boxes: [
      { pos: [-2.8, 0.6, 1.6], size: [1.6, 1.6], color: '#ffb46b', intensity: 9 },
      { pos: [2.8, 1.4, -1.0], size: [1.2, 2.0], color: '#6d89c7', intensity: 2.2 },
    ],
    key: { dir: [-0.8, 0.25, 0.55], color: '#ffc488', intensity: 2.0, soft: 5 },
    fill: { color: '#5a6e9a', intensity: 0.12 },
    exposure: 1.05,
  },
}

function roomScene(def: PresetDef): THREE.Scene {
  const scene = new THREE.Scene()
  const geo = new THREE.SphereGeometry(10, 48, 24)
  const top = new THREE.Color(def.bg[0])
  const bot = new THREE.Color(def.bg[1])
  const colors: number[] = []
  const pos = geo.attributes.position
  for (let i = 0; i < pos.count; i++) {
    const t = THREE.MathUtils.smoothstep(pos.getY(i), -8, 8)
    const c = bot.clone().lerp(top, t)
    colors.push(c.r, c.g, c.b)
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  scene.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })))
  for (const b of def.boxes) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(b.size[0], b.size[1]),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(b.color).multiplyScalar(b.intensity), side: THREE.DoubleSide }))
    m.position.set(...b.pos)
    m.lookAt(0, 0, 0)
    scene.add(m)
  }
  return scene
}

export class LightRig {
  readonly group = new THREE.Group()
  readonly key: THREE.DirectionalLight
  readonly fill: THREE.HemisphereLight
  private envCache = new Map<LightingPreset, THREE.Texture>()
  preset: LightingPreset = 'studio'
  rotation = 0 // radians about the vertical axis

  constructor(private renderer: THREE.WebGLRenderer, private scene: THREE.Scene, private target: THREE.Vector3) {
    this.key = new THREE.DirectionalLight(0xffffff, 2)
    this.key.castShadow = true
    this.key.shadow.mapSize.set(2048, 2048)
    this.key.shadow.bias = -0.0006
    this.key.shadow.normalBias = 0.4
    this.key.shadow.blurSamples = 16
    const cam = this.key.shadow.camera as THREE.OrthographicCamera
    cam.left = -140; cam.right = 140; cam.top = 140; cam.bottom = -140; cam.near = 1; cam.far = 1200
    this.fill = new THREE.HemisphereLight(0xffffff, 0x404040, 0.2)
    this.group.add(this.key, this.key.target, this.fill)
    scene.add(this.group)
  }

  env(preset: LightingPreset): THREE.Texture {
    let t = this.envCache.get(preset)
    if (!t) {
      const pm = new THREE.PMREMGenerator(this.renderer)
      t = pm.fromScene(roomScene(PRESETS[preset]), 0.02).texture
      pm.dispose()
      this.envCache.set(preset, t)
    }
    return t
  }

  apply(preset: LightingPreset, rotation = this.rotation) {
    this.preset = preset
    this.rotation = rotation
    const def = PRESETS[preset]
    this.scene.environment = this.env(preset)
    this.scene.environmentRotation.set(0, rotation, 0)
    const d = new THREE.Vector3(...def.key.dir).normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), rotation)
    this.key.position.copy(this.target).addScaledVector(d, 500)
    this.key.target.position.copy(this.target)
    this.key.color.set(def.key.color)
    this.key.intensity = def.key.intensity
    this.key.shadow.radius = def.key.soft * 3
    this.fill.color.set(def.fill.color)
    this.fill.intensity = def.fill.intensity
    this.renderer.toneMappingExposure = def.exposure
  }

  background(preset: LightingPreset): THREE.Color {
    return new THREE.Color(PRESETS[preset].bg[1]).lerp(new THREE.Color(PRESETS[preset].bg[0]), 0.35)
  }

  dispose() {
    this.envCache.forEach((t) => t.dispose())
  }
}
