import type { PlanSummary } from '../api'

// Mid-sagittal soft-tissue profile: current outline, predicted landmark positions with 90% bands,
// plus E-line and zero-meridian references.

interface Props {
  profile: [number, number][] | undefined   // (y, z) head frame
  bone?: [number, number][] | undefined
  landmarks: Record<string, { position: [number, number, number] }>
  plan: PlanSummary | null
  tIndex: number
}

const KEYS = ['subnasale', 'labrale_superius', 'stomion', 'labrale_inferius', 'soft_b_point', 'soft_pogonion',
  'soft_gnathion', 'soft_menton']

export default function ProfileChart({ profile, bone, landmarks, plan, tIndex }: Props) {
  if (!profile || !landmarks.soft_pogonion) return null
  const pg = landmarks.soft_pogonion.position
  const sn = landmarks.subnasale?.position ?? pg
  // window: lower face
  const zmax = sn[2] + 18, zmin = (landmarks.soft_menton?.position[2] ?? pg[2] - 15) - 18
  const ymax = Math.max(pg[1], landmarks.pronasale?.position[1] ?? pg[1]) + 10
  const ymin = ymax - (zmax - zmin) * 0.75
  const W = 300, H = 400
  const sx = (y: number) => ((y - ymin) / (ymax - ymin)) * W
  const sy = (z: number) => H - ((z - zmin) / (zmax - zmin)) * H
  const inWin = (p: [number, number]) => p[1] > zmin - 5 && p[1] < zmax + 5 && p[0] > ymin - 5
  const segs: string[] = []
  let cur: string[] = []
  for (const p of profile) {
    if (inWin(p)) cur.push(`${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`)
    else if (cur.length) { segs.push(cur.join(' ')); cur = [] }
  }
  if (cur.length) segs.push(cur.join(' '))
  const boneSegs: string[] = []
  if (bone) boneSegs.push(bone.filter(inWin).map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' '))
  const tp = plan?.timepoints[Math.round(tIndex)]
  const pred = KEYS.filter((k) => tp?.landmarks[k])
  const predLine = pred.map((k) => `${sx(tp!.landmarks[k].median[1]).toFixed(1)},${sy(tp!.landmarks[k].median[2]).toFixed(1)}`).join(' ')
  const prn = landmarks.pronasale?.position
  const n = landmarks.soft_nasion?.position
  const pgPost = tp?.landmarks.soft_pogonion?.median
  return (
    <svg className="profile" viewBox={`0 0 ${W} ${H}`}>
      <rect x={0} y={0} width={W} height={H} className="bg" />
      {n && <line x1={sx(n[1])} x2={sx(n[1])} y1={0} y2={H} className="ref" />}
      {n && <text x={sx(n[1]) + 3} y={12} className="reflabel">zero meridian</text>}
      {prn && <line x1={sx(prn[1])} y1={sy(prn[2])} x2={sx(pg[1])} y2={sy(pg[2])} className="eline pre" />}
      {prn && pgPost && <line x1={sx(prn[1])} y1={sy(prn[2])} x2={sx(pgPost[1])} y2={sy(pgPost[2])} className="eline post" />}
      {boneSegs.map((s, i) => <polyline key={'b' + i} points={s} className="bone" />)}
      {segs.map((s, i) => <polyline key={i} points={s} className="pre" />)}
      {tp && pred.map((k) => {
        const L = tp.landmarks[k]
        return (
          <g key={k}>
            <line x1={sx(L.p05[1])} x2={sx(L.p95[1])} y1={sy(L.median[2])} y2={sy(L.median[2])} className="band" />
            <line x1={sx(L.median[1])} x2={sx(L.median[1])} y1={sy(L.p05[2])} y2={sy(L.p95[2])} className="band" />
          </g>
        )
      })}
      {tp && <polyline points={predLine} className="post" />}
      {tp && pred.map((k) => <circle key={k} cx={sx(tp.landmarks[k].median[1])} cy={sy(tp.landmarks[k].median[2])} r={2.5} className="pt" />)}
      <g className="legend" transform={`translate(8, ${H - 44})`}>
        <line x1={0} x2={18} y1={0} y2={0} className="pre" /><text x={22} y={4}>current</text>
        <line x1={0} x2={18} y1={14} y2={14} className="post" /><text x={22} y={18}>predicted (median)</text>
        <line x1={0} x2={18} y1={28} y2={28} className="band" /><text x={22} y={32}>90% range</text>
      </g>
    </svg>
  )
}
