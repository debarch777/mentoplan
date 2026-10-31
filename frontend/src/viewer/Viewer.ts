import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { GTAOPass } from 'three/examples/jsm/postprocessing/GTAOPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { LightRig, LightingPreset } from './lighting'
import { makeSkinMaterial, SkinParams, updateSkinParams } from './skinMaterial'

// Patient head frame (RAS, mm: +x right, +y anterior, +z superior) -> three.js (x: patient left,
// y: up, z: towards the viewer when looking at the face).  Proper rotation (det = +1).
export const HEAD_TO_THREE = new THREE.Matrix4().set(
  -1, 0, 0, 0,
  0, 0, 1, 0,
  0, 1, 0, 0,
  0, 0, 0, 1,
)

export function headToThree(p: [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(p[0], p[1], p[2]).applyMatrix4(HEAD_TO_THREE)
}

export type ImplantMaterialName = 'silicone' | 'porous_polyethylene' | 'peek' | 'titanium'

function implantMaterial(name: ImplantMaterialName): THREE.MeshPhysicalMaterial {
  switch (name) {
    case 'porous_polyethylene':
      return new THREE.MeshPhysicalMaterial({ color: '#f3f1ec', roughness: 0.92, sheen: 0.3, sheenColor: new THREE.Color('#ffffff') })
    case 'peek':
      return new THREE.MeshPhysicalMaterial({ color: '#d8c7a3', roughness: 0.38, clearcoat: 0.3 })
    case 'titanium':
      return new THREE.MeshPhysicalMaterial({ color: '#b9bcc0', metalness: 1, roughness: 0.32 })
    default:
      return new THREE.MeshPhysicalMaterial({
        color: '#dfe9ec', roughness: 0.22, transmission: 0.55, thickness: 4, ior: 1.41,
        attenuationColor: new THREE.Color('#cfe0e6'), attenuationDistance: 12, clearcoat: 0.5,
      })
  }
}

export interface Layers {
  skinOpacity: number
  bone: boolean
  implant: boolean
  landmarks: boolean
}

export class Viewer {
  readonly renderer: THREE.WebGLRenderer
  readonly scene = new THREE.Scene()
  readonly root = new THREE.Group()
  readonly rig: LightRig
  private composer: EffectComposer
  private gtao: GTAOPass
  skin?: THREE.Mesh
  skinMat: THREE.MeshPhysicalMaterial
  private boneMeshes: THREE.Mesh[] = []
  private implantMesh?: THREE.Mesh
  private landmarkGroup = new THREE.Group()
  readonly jawGroup = new THREE.Group()   // mandible + lower teeth + implant (moves with jaw opening)
  private boneMat = new THREE.MeshPhysicalMaterial({ color: '#e9dfcc', roughness: 0.62, sheen: 0.2, sheenColor: new THREE.Color('#fff8ea') })
  private teethMat = new THREE.MeshPhysicalMaterial({ color: '#f4f0e6', roughness: 0.25, clearcoat: 0.6 })
  private width = 1
  private height = 1
  layers: Layers = { skinOpacity: 1, bone: false, implant: false, landmarks: false }

  constructor(readonly canvas: HTMLCanvasElement, readonly camera: THREE.PerspectiveCamera, target: THREE.Vector3,
              skinParams: SkinParams) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true, alpha: false })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.VSMShadowMap   // soft penumbrae (area-light look)
    this.renderer.toneMapping = THREE.AgXToneMapping
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.root.matrixAutoUpdate = false
    this.root.matrix.copy(HEAD_TO_THREE)
    this.scene.add(this.root)
    this.root.add(this.jawGroup)
    this.root.add(this.landmarkGroup)
    this.rig = new LightRig(this.renderer, this.scene, target)
    this.skinMat = makeSkinMaterial(skinParams)
    const rt = new THREE.WebGLRenderTarget(1, 1, { samples: 4, type: THREE.HalfFloatType })
    this.composer = new EffectComposer(this.renderer, rt)
    this.composer.addPass(new RenderPass(this.scene, camera))
    this.gtao = new GTAOPass(this.scene, camera, 1, 1)
    this.gtao.blendIntensity = 0.85
    this.gtao.updateGtaoMaterial({ radius: 6, distanceExponent: 1.5, thickness: 4, scale: 1, samples: 16 })
    this.composer.addPass(this.gtao)
    this.composer.addPass(new OutputPass())
  }

  setLighting(preset: LightingPreset, rotation: number) {
    this.rig.apply(preset, rotation)
    this.scene.background = this.rig.background(preset)
  }

  setSkinGeometry(geom: THREE.BufferGeometry) {
    if (this.skin) {
      this.root.remove(this.skin)
    }
    this.skin = new THREE.Mesh(geom, this.skinMat)
    this.skin.castShadow = true
    this.skin.receiveShadow = true
    this.skin.renderOrder = 2
    this.root.add(this.skin)
    this.applyLayers()
  }

  private teethMeshes: THREE.Mesh[] = []

  setBones(cranium?: THREE.BufferGeometry, mandible?: THREE.BufferGeometry, teeth?: THREE.BufferGeometry,
           teethUpper?: THREE.BufferGeometry) {
    for (const m of [...this.boneMeshes, ...this.teethMeshes]) m.parent?.remove(m)
    this.boneMeshes = []
    this.teethMeshes = []
    if (cranium) {
      const m = new THREE.Mesh(cranium, this.boneMat)
      m.castShadow = m.receiveShadow = true
      this.root.add(m)
      this.boneMeshes.push(m)
    }
    if (mandible) {
      const m = new THREE.Mesh(mandible, this.boneMat)
      m.castShadow = m.receiveShadow = true
      this.jawGroup.add(m)
      this.boneMeshes.push(m)
    }
    // dentition is always rendered: it becomes visible when an expression opens the lips
    if (teeth) {
      const m = new THREE.Mesh(teeth, this.teethMat)
      this.jawGroup.add(m)
      this.teethMeshes.push(m)
    }
    if (teethUpper) {
      const m = new THREE.Mesh(teethUpper, this.teethMat)
      this.root.add(m)
      this.teethMeshes.push(m)
    }
    this.applyLayers()
  }

  setImplant(geom: THREE.BufferGeometry | null, material: ImplantMaterialName = 'silicone') {
    if (this.implantMesh) {
      this.jawGroup.remove(this.implantMesh)
      ;(this.implantMesh.material as THREE.Material).dispose()
      this.implantMesh = undefined
    }
    if (geom) {
      this.implantMesh = new THREE.Mesh(geom, implantMaterial(material))
      this.implantMesh.castShadow = true
      this.implantMesh.renderOrder = 1
      this.jawGroup.add(this.implantMesh)
    }
    this.applyLayers()
  }

  setLandmarks(points: { position: [number, number, number]; kind: 'soft' | 'bone' | 'nerve'; confidence: number }[]) {
    this.landmarkGroup.clear()
    const geo = new THREE.SphereGeometry(1.1, 16, 12)
    const cols = { soft: '#3ad1c5', bone: '#f2c14e', nerve: '#ff4d4d' }
    for (const p of points) {
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: cols[p.kind], depthTest: false, transparent: true,
        opacity: 0.5 + 0.5 * p.confidence }))
      m.position.set(...p.position)
      m.renderOrder = 10
      this.landmarkGroup.add(m)
    }
    this.applyLayers()
  }

  setJaw(matrix: THREE.Matrix4) {
    this.jawGroup.matrixAutoUpdate = false
    this.jawGroup.matrix.copy(matrix)
    this.jawGroup.matrixWorldNeedsUpdate = true
  }

  setLayers(l: Partial<Layers>) {
    this.layers = { ...this.layers, ...l }
    this.applyLayers()
  }

  private applyLayers() {
    const op = this.layers.skinOpacity
    this.skinMat.transparent = op < 0.999
    this.skinMat.opacity = op
    this.skinMat.depthWrite = op >= 0.999
    this.skinMat.needsUpdate = true
    if (this.skin) this.skin.visible = op > 0.01
    for (const m of this.boneMeshes) m.visible = this.layers.bone
    if (this.implantMesh) this.implantMesh.visible = this.layers.implant
    this.landmarkGroup.visible = this.layers.landmarks
  }

  setSkin(p: Partial<SkinParams>) {
    updateSkinParams(this.skinMat, p)
  }

  resize(w: number, h: number) {
    if (w < 2 || h < 2) return
    this.width = w
    this.height = h
    this.renderer.setSize(w, h, false)
    this.composer.setSize(w, h)
    this.gtao.setSize(w * this.renderer.getPixelRatio(), h * this.renderer.getPixelRatio())
  }

  render() {
    this.camera.aspect = this.width / this.height
    this.camera.updateProjectionMatrix()
    this.composer.render()
  }

  snapshot(): string {
    this.render()
    return this.renderer.domElement.toDataURL('image/png')
  }

  dispose() {
    this.rig.dispose()
    this.composer.dispose()
    this.renderer.dispose()
  }
}
