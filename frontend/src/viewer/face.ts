import * as THREE from 'three'
import type { Bundle, Landmark } from '../api'

// Skin surface state: base geometry (head frame) plus the deformation fields delivered by the
// planning service, composed on the CPU into pre-op and post-op geometries.

export interface PlanFields {
  active: Uint32Array
  disp: Float32Array[]   // per timepoint, active x 3
  low: Float32Array[]    // per timepoint, normal offset of the 5th percentile surface (<= 0)
  high: Float32Array[]   // per timepoint, normal offset of the 95th percentile surface (>= 0)
}

export interface ExpressionFields {
  active: Uint32Array
  fields: Record<string, Float32Array>       // active x 3, unit activation
  meta: Record<string, { label: string; jaw_deg?: number; axis_point?: number[]; axis_dir?: number[] }>
}

export interface AgingFields {
  active: Uint32Array
  perDecade: Float32Array                    // active x 3 displacement per 10 years
  meta: Record<string, unknown>
}

export interface DeformState {
  timeline: number      // continuous index into timepoints
  range: number         // -1 (5th pct) .. 0 (likely) .. +1 (95th pct)
  expression: Record<string, number>
  agingYears: number
}

export class FaceSurface {
  readonly n: number
  readonly base: Float32Array
  readonly baseNormal: Float32Array
  readonly index: Uint32Array
  readonly lip: Float32Array
  readonly beard: Float32Array
  readonly interior: Float32Array      // oral-cavity side of the sealed mouth surface
  readonly albedo: Float32Array        // linear RGB from a registered face scan
  readonly texMask: Float32Array       // 1 where the face scan provides colour
  private side: Int8Array              // +1 upper lip / -1 lower lip near the fissure
  readonly pre: THREE.BufferGeometry
  readonly post: THREE.BufferGeometry
  private cut: Float32Array
  private mouth: Float32Array
  plan?: PlanFields
  expr?: ExpressionFields
  aging?: AgingFields

  constructor(mesh: Bundle, lms: Record<string, Landmark>, fissureHalfWidth = 22, private fissureZ?: number) {
    this.base = mesh.arrays.position as Float32Array
    this.baseNormal = mesh.arrays.normal as Float32Array
    this.index = mesh.arrays.index as Uint32Array
    this.n = this.base.length / 3
    this.lip = new Float32Array(this.n)
    this.beard = new Float32Array(this.n)
    this.interior = new Float32Array(this.n)
    this.albedo = new Float32Array(this.n * 3)
    this.texMask = new Float32Array(this.n)
    const ar = mesh.arrays.attr_albedo_r, ag = mesh.arrays.attr_albedo_g, ab = mesh.arrays.attr_albedo_b
    const am = mesh.arrays.attr_albedo_mask
    if (ar && ag && ab && am) {
      for (let i = 0; i < this.n; i++) {
        this.albedo[3 * i] = ar[i]; this.albedo[3 * i + 1] = ag[i]; this.albedo[3 * i + 2] = ab[i]
        this.texMask[i] = am[i]
      }
    }
    // flat caps where the scan field of view cuts through the head are not skin: hidden in rendering
    this.cut = new Float32Array(this.n)
    this.mouth = new Float32Array(this.n)
    const fd = mesh.arrays.attr_fov_dist
    if (fd) for (let i = 0; i < this.n; i++) this.cut[i] = 1 - THREE.MathUtils.smoothstep(fd[i], 0.6, 1.6)
    this.side = new Int8Array(this.n)
    this.computeMasks(lms, fissureHalfWidth)
    this.computeMouth(lms, fissureHalfWidth)
    this.pre = this.makeGeometry()
    this.post = this.makeGeometry()
  }

  makeGeometry() {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.base), 3).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(this.baseNormal), 3).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aUncert', new THREE.BufferAttribute(new Float32Array(this.n), 1).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aChange', new THREE.BufferAttribute(new Float32Array(this.n), 1).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aLip', new THREE.BufferAttribute(this.lip, 1))
    g.setAttribute('aBeard', new THREE.BufferAttribute(this.beard, 1))
    g.setAttribute('aMouth', new THREE.BufferAttribute(this.mouth, 1))
    g.setAttribute('aAlbedo', new THREE.BufferAttribute(this.albedo, 3))
    g.setAttribute('aTexMask', new THREE.BufferAttribute(this.texMask, 1))
    g.setAttribute('aCavity', new THREE.BufferAttribute(new Float32Array(this.cut), 1).setUsage(THREE.DynamicDrawUsage))
    g.setIndex(new THREE.BufferAttribute(this.index, 1))
    g.computeBoundingSphere()
    return g
  }

  /** Vermilion (lip) and beard-area masks from the landmarks (smooth, 0..1). */
  private computeMasks(L: Record<string, Landmark>, hw: number) {
    const g = (k: string) => (L[k] ? L[k].position : null)
    const sto = g('stomion'), ls = g('labrale_superius'), li = g('labrale_inferius'), sn = g('subnasale'),
      me = g('soft_menton')
    if (sto) for (let i = 0; i < this.n; i++) {
      // back faces near the mouth are seen through parted lips (oral mucosa); elsewhere through a
      // scan-field-of-view cut (neutral dark interior)
      const d = Math.hypot(this.base[3 * i] - sto[0], this.base[3 * i + 1] - sto[1], this.base[3 * i + 2] - sto[2])
      this.mouth[i] = 1 - THREE.MathUtils.smoothstep(d, 32, 48)
    }
    if (!sto || !ls || !li) return
    // vermilion height from the landmarks, bounded by anatomy (upper 4-9 mm, lower 5-14 mm): the
    // most anterior upper-lip point may sit right under the nose in a retrusive lip
    const upH = THREE.MathUtils.clamp(ls[2] - sto[2] + 2.2, 4, 9)
    const loH = THREE.MathUtils.clamp(sto[2] - li[2] + 3.0, 5, 14)
    for (let i = 0; i < this.n; i++) {
      const x = this.base[3 * i], y = this.base[3 * i + 1], z = this.base[3 * i + 2]
      const ny = this.baseNormal[3 * i + 1]
      const dz = z - sto[2]
      const h = dz >= 0 ? upH * (1 - 0.18 * Math.exp(-((x / 4) ** 2))) : loH   // slight cupid's bow dip
      const e = (x / (hw * 1.02)) ** 2 + (dz / h) ** 2
      const front = THREE.MathUtils.smoothstep(ny, -0.1, 0.35) * (y > sto[1] - 14 ? 1 : 0)
      this.lip[i] = THREE.MathUtils.clamp((1.0 - e) / 0.25, 0, 1) * front
      if (sn && me) {
        const inZ = THREE.MathUtils.smoothstep(z, me[2] - 22, me[2] - 8) * (1 - THREE.MathUtils.smoothstep(z, sn[2] - 6, sn[2] - 1))
        const inX = 1 - THREE.MathUtils.smoothstep(Math.abs(x), 48, 62)
        const fwd = THREE.MathUtils.smoothstep(y, sto[1] - 75, sto[1] - 55)
        this.beard[i] = inZ * inX * fwd * (1 - this.lip[i])
      }
    }
  }

  /** Vertices of the surface that seals the mouth opening (between and behind the lips). */
  private computeMouth(L: Record<string, Landmark>, hw: number) {
    const sto = L.stomion?.position, tip = L.lower_incisor_tip?.position
    if (!sto) return
    const zc = this.fissureZ ?? sto[2]
    const yBack = (tip ? tip[1] : sto[1] - 14) - 8
    for (let i = 0; i < this.n; i++) {
      const x = this.base[3 * i], y = this.base[3 * i + 1], z = this.base[3 * i + 2]
      const inZone = Math.abs(x) < hw + 3 && Math.abs(z - zc) < 12 && y > yBack
      if (!inZone) continue
      this.side[i] = z >= zc ? 1 : -1
      const lat = 1 - THREE.MathUtils.smoothstep(Math.abs(x), hw - 3, hw + 1)
      const depth = THREE.MathUtils.smoothstep(sto[1] - y, 1.5, 3.5)
      const near = 1 - THREE.MathUtils.smoothstep(Math.abs(z - zc), 6, 9)
      // everything more than ~4 mm behind the lip front is the surface sealing the mouth opening
      // (segmentation closing) or oral mucosa: hidden as soon as the lips part
      const deep = THREE.MathUtils.smoothstep(sto[1] - y, 3.0, 4.5)
      const ny = this.baseNormal[3 * i + 1]
      const facing = 1 - THREE.MathUtils.smoothstep(ny, 0.35, 0.7)
      this.interior[i] = Math.max(lat * near * Math.max(depth * facing, deep),
        lat * (1 - THREE.MathUtils.smoothstep(Math.abs(z - zc), 0.6, 1.4)) * THREE.MathUtils.smoothstep(sto[1] - y, -2, 0.5))
    }
  }

  setPlan(b: Bundle | null) {
    this.plan = b ? planFieldsFromBundle(b) : undefined
  }

  setExpressions(b: Bundle) {
    const fields: Record<string, Float32Array> = {}
    for (const k of Object.keys(b.arrays)) if (k.startsWith('expr_')) fields[k.slice(5)] = b.arrays[k] as Float32Array
    this.expr = { active: b.arrays.active as Uint32Array, fields, meta: b.meta.expressions }
  }

  setAging(b: Bundle) {
    this.aging = { active: b.arrays.active as Uint32Array, perDecade: b.arrays.age_decade as Float32Array, meta: b.meta.aging }
  }

  /** Recompute both geometries for the given state. */
  update(s: DeformState, overlayMode: 0 | 1 | 2) {
    const pre = this.pre.attributes.position.array as Float32Array
    pre.set(this.base)
    // shared terms: expression + aging (apply to both pre- and post-op)
    if (this.expr) {
      const { active, fields } = this.expr
      for (const [k, w] of Object.entries(s.expression)) {
        const f = fields[k]
        if (!f || Math.abs(w) < 1e-3) continue
        for (let j = 0; j < active.length; j++) {
          const i = active[j]
          pre[3 * i] += w * f[3 * j]; pre[3 * i + 1] += w * f[3 * j + 1]; pre[3 * i + 2] += w * f[3 * j + 2]
        }
      }
    }
    if (this.aging && s.agingYears > 0) {
      const { active, perDecade } = this.aging
      const w = s.agingYears / 10
      for (let j = 0; j < active.length; j++) {
        const i = active[j]
        pre[3 * i] += w * perDecade[3 * j]; pre[3 * i + 1] += w * perDecade[3 * j + 1]; pre[3 * i + 2] += w * perDecade[3 * j + 2]
      }
    }
    // lip separation across the oral fissure (expressions / jaw opening): opens the oral cavity
    let up = 0, nu = 0, lo = 0, nl = 0
    for (let i = 0; i < this.n; i++) {
      const sd = this.side[i]
      if (!sd || this.interior[i] > 0.2) continue
      const dz = pre[3 * i + 2] - this.base[3 * i + 2]
      if (sd > 0) { up += dz; nu++ } else { lo += dz; nl++ }
    }
    const gap = (nu ? up / nu : 0) - (nl ? lo / nl : 0)
    const open = THREE.MathUtils.smoothstep(gap, 0.6, 2.5)
    for (const g of [this.pre, this.post]) {
      const cav = g.attributes.aCavity.array as Float32Array
      for (let i = 0; i < this.n; i++) cav[i] = Math.max(this.interior[i] * open, this.cut[i])
      g.attributes.aCavity.needsUpdate = true
    }
    this.composePlan(this.post, this.plan, s)
    this.finish(this.pre)
    void overlayMode
  }

  /** A separate post-operative geometry (variant comparison); call updateVariant after update(). */
  variantGeometry(): THREE.BufferGeometry {
    return this.makeGeometry()
  }

  updateVariant(g: THREE.BufferGeometry, plan: PlanFields | undefined, s: DeformState) {
    ;(g.attributes.aCavity.array as Float32Array).set(this.pre.attributes.aCavity.array as Float32Array)
    g.attributes.aCavity.needsUpdate = true
    this.composePlan(g, plan, s)
  }

  /** post = current (incl. expression / aging) + predicted change at the timeline / range position. */
  private composePlan(g: THREE.BufferGeometry, plan: PlanFields | undefined, s: DeformState) {
    const post = g.attributes.position.array as Float32Array
    post.set(this.pre.attributes.position.array as Float32Array)
    const unc = g.attributes.aUncert.array as Float32Array
    const chg = g.attributes.aChange.array as Float32Array
    unc.fill(0)
    chg.fill(0)
    if (plan) {
      const { active, disp, low, high } = plan
      const T = disp.length
      const t = THREE.MathUtils.clamp(s.timeline, 0, T - 1)
      const i0 = Math.floor(t), i1 = Math.min(T - 1, i0 + 1), f = t - i0
      const d0 = disp[i0], d1 = disp[i1], l0 = low[i0], l1 = low[i1], h0 = high[i0], h1 = high[i1]
      const nrm = this.baseNormal
      for (let j = 0; j < active.length; j++) {
        const i = active[j]
        const dx = d0[3 * j] * (1 - f) + d1[3 * j] * f
        const dy = d0[3 * j + 1] * (1 - f) + d1[3 * j + 1] * f
        const dz = d0[3 * j + 2] * (1 - f) + d1[3 * j + 2] * f
        const lo = l0[j] * (1 - f) + l1[j] * f
        const hi = h0[j] * (1 - f) + h1[j] * f
        const off = s.range < 0 ? -s.range * lo : s.range * hi
        const nx = nrm[3 * i], ny = nrm[3 * i + 1], nz = nrm[3 * i + 2]
        post[3 * i] += dx + off * nx
        post[3 * i + 1] += dy + off * ny
        post[3 * i + 2] += dz + off * nz
        unc[i] = hi - lo
        chg[i] = dx * nx + dy * ny + dz * nz
      }
    }
    g.attributes.aUncert.needsUpdate = true
    g.attributes.aChange.needsUpdate = true
    this.finish(g)
  }

  private adj?: { ptr: Int32Array; idx: Int32Array }

  /** Vertex adjacency (CSR) from the triangle list, built once. */
  private adjacency() {
    if (this.adj) return this.adj
    const n = this.n, f = this.index
    const deg = new Int32Array(n + 1)
    for (let t = 0; t < f.length; t += 3) { deg[f[t]] += 2; deg[f[t + 1]] += 2; deg[f[t + 2]] += 2 }
    const ptr = new Int32Array(n + 1)
    for (let i = 0; i < n; i++) ptr[i + 1] = ptr[i] + deg[i]
    const fill = ptr.slice(0, n)
    const idx = new Int32Array(ptr[n])
    for (let t = 0; t < f.length; t += 3) {
      const a = f[t], b = f[t + 1], c = f[t + 2]
      idx[fill[a]++] = b; idx[fill[a]++] = c
      idx[fill[b]++] = a; idx[fill[b]++] = c
      idx[fill[c]++] = a; idx[fill[c]++] = b
    }
    this.adj = { ptr, idx }
    return this.adj
  }

  /** Positions are kept exactly; shading normals are lightly diffused over the surface so that
   *  sub-millimetre CBCT banding does not read as stripes under grazing light. */
  private finish(g: THREE.BufferGeometry, iterations = 6) {
    g.attributes.position.needsUpdate = true
    g.computeVertexNormals()
    const nrm = g.attributes.normal.array as Float32Array
    const { ptr, idx } = this.adjacency()
    const tmp = new Float32Array(nrm.length)
    for (let it = 0; it < iterations; it++) {
      for (let i = 0; i < this.n; i++) {
        let x = nrm[3 * i], y = nrm[3 * i + 1], z = nrm[3 * i + 2]
        const k0 = ptr[i], k1 = ptr[i + 1]
        let sx = 0, sy = 0, sz = 0
        for (let k = k0; k < k1; k++) { const j = idx[k]; sx += nrm[3 * j]; sy += nrm[3 * j + 1]; sz += nrm[3 * j + 2] }
        const w = k1 > k0 ? 0.5 / (k1 - k0) : 0
        x = 0.5 * x + w * sx; y = 0.5 * y + w * sy; z = 0.5 * z + w * sz
        const l = Math.hypot(x, y, z) || 1
        tmp[3 * i] = x / l; tmp[3 * i + 1] = y / l; tmp[3 * i + 2] = z / l
      }
      nrm.set(tmp)
    }
    g.attributes.normal.needsUpdate = true
    g.computeBoundingSphere()
  }

  /** Rigid jaw transform (head frame) for the current expression weights. */
  jawMatrix(s: DeformState): THREE.Matrix4 {
    const m = new THREE.Matrix4()
    if (!this.expr) return m
    let angle = 0
    let point: number[] | undefined, dir: number[] | undefined
    for (const [k, w] of Object.entries(s.expression)) {
      const meta = this.expr.meta[k]
      if (meta?.jaw_deg && meta.axis_point && meta.axis_dir) {
        angle += w * meta.jaw_deg
        point = meta.axis_point
        dir = meta.axis_dir
      }
    }
    if (!point || !dir || Math.abs(angle) < 1e-4) return m
    const p = new THREE.Vector3(...point)
    const axis = new THREE.Vector3(...dir).normalize()
    m.makeTranslation(p.x, p.y, p.z)
      .multiply(new THREE.Matrix4().makeRotationAxis(axis, THREE.MathUtils.degToRad(angle)))
      .multiply(new THREE.Matrix4().makeTranslation(-p.x, -p.y, -p.z))
    return m
  }
}

export function planFieldsFromBundle(b: Bundle): PlanFields {
  const T = b.meta.n_timepoints as number
  const disp: Float32Array[] = [], low: Float32Array[] = [], high: Float32Array[] = []
  for (let i = 0; i < T; i++) {
    disp.push(b.arrays[`disp_${i}`] as Float32Array)
    low.push(b.arrays[`low_${i}`] as Float32Array)
    high.push(b.arrays[`high_${i}`] as Float32Array)
  }
  return { active: b.arrays.active as Uint32Array, disp, low, high }
}

export function bundleGeometry(b: Bundle): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(b.arrays.position as Float32Array, 3))
  g.setAttribute('normal', new THREE.BufferAttribute(b.arrays.normal as Float32Array, 3))
  g.setIndex(new THREE.BufferAttribute(b.arrays.index as Uint32Array, 1))
  g.computeBoundingSphere()
  return g
}
