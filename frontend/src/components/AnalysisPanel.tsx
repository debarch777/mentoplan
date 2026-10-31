import type { MetricRow, PlanSummary, Risk } from '../api'

function fmt(v: number | undefined | null, d = 1) {
  return v === undefined || v === null || Number.isNaN(v) ? '–' : v.toFixed(d)
}

function statusClass(s: string) {
  return s === 'within' ? 'ok' : s === 'n/a' ? '' : 'warn'
}

export function MetricsTable({ pre, plan, tIndex }: { pre: MetricRow[]; plan: PlanSummary | null; tIndex: number }) {
  const post = plan?.timepoints[Math.round(tIndex)]?.metrics ?? []
  const byKey = Object.fromEntries(post.map((r) => [r.key, r]))
  return (
    <table className="metrics">
      <thead>
        <tr><th>Measurement</th><th>Now</th><th>Predicted (90% range)</th><th>Norm</th></tr>
      </thead>
      <tbody>
        {pre.map((r) => {
          const q = byKey[r.key]
          return (
            <tr key={r.key} title={r.reference + (r.depends_on_head_pose ? ' · depends on head orientation' : '')}>
              <td>{r.label}{r.depends_on_head_pose && <sup>†</sup>}</td>
              <td className={'mono ' + statusClass(r.status)}>{fmt(r.value)}{r.unit === 'deg' ? '°' : ''}</td>
              <td className={'mono ' + (q ? statusClass(q.status) : '')}>
                {q ? <>{fmt(q.value)}{r.unit === 'deg' ? '°' : ''}{q.p05 !== undefined && <span className="range"> [{fmt(q.p05)}, {fmt(q.p95)}]</span>}</> : '–'}
              </td>
              <td className="mono dim">{r.norm[0] !== null ? `${fmt(r.norm[0], 0)}…${fmt(r.norm[1], 0)}` : ''}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export function LandmarkChanges({ plan, tIndex }: { plan: PlanSummary | null; tIndex: number }) {
  if (!plan) return null
  const tp = plan.timepoints[Math.round(tIndex)]
  const names: [string, string][] = [['soft_pogonion', "Soft pogonion"], ['soft_gnathion', "Soft gnathion"], ['soft_menton', "Soft menton"],
    ['soft_b_point', 'Labiomental fold'], ['labrale_inferius', 'Lower lip'], ['stomion', 'Stomion']]
  return (
    <table className="metrics">
      <thead><tr><th>Point</th><th>Forward (mm)</th><th>Vertical (mm)</th></tr></thead>
      <tbody>
        {names.filter(([k]) => tp.landmarks[k]).map(([k, label]) => {
          const L = tp.landmarks[k]
          const d = (i: number, w: 'median' | 'p05' | 'p95') => L[w][i] - L.pre[i]
          return (
            <tr key={k}>
              <td>{label}</td>
              <td className="mono">{fmt(d(1, 'median'))}<span className="range"> [{fmt(d(1, 'p05'))}, {fmt(d(1, 'p95'))}]</span></td>
              <td className="mono">{fmt(d(2, 'median'))}<span className="range"> [{fmt(d(2, 'p05'))}, {fmt(d(2, 'p95'))}]</span></td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export function CalibrationBox({ plan }: { plan: PlanSummary | null }) {
  const c = plan?.timepoints[plan.timepoints.length - 1]?.calibration as any
  if (!c || c.ratio_fem === undefined) return null
  return (
    <div className="calib">
      <div className="calib-title">How the projection estimate was formed</div>
      <div className="calib-row"><span>Physics model (your tissues)</span><span className="mono">{fmt(c.ratio_fem, 2)} ± {fmt(c.ratio_fem_sd, 2)}</span></div>
      <div className="calib-row"><span>Clinical evidence for {fmt(c.soft_tissue_thickness_mm)} mm chin tissue</span><span className="mono">{fmt(c.ratio_evidence, 2)} ± {fmt(c.ratio_evidence_sd, 2)}</span></div>
      <div className="calib-row strong"><span>Combined soft : implant ratio</span><span className="mono">{fmt(c.ratio_posterior, 2)} ± {fmt(c.ratio_posterior_sd, 2)}</span></div>
      <div className="calib-note">Soft-tissue pogonion advance per mm of implant at the bony pogonion, settled result (12 months, incl. bone settling).</div>
      {plan && plan.timepoints[0]?.posture_sd_mm !== undefined && (
        <div className="calib-note">Ranges show the outcome of the surgery. In addition, the resting position of the lips and chin
          varies between occasions by about ±{fmt(1.645 * plan.timepoints[0].posture_sd_mm)} mm (90%, {(plan as any).uncertainty_profile ?? 'validated'} profile).</div>
      )}
    </div>
  )
}

export function RiskPanel({ plan }: { plan: PlanSummary | null }) {
  if (!plan) return <div className="empty">Run a prediction to see the risk assessment.</div>
  return (
    <div className="risks">
      <div className={'overall ' + plan.overall_risk}>Overall: {plan.overall_risk} risk</div>
      {plan.warnings.map((w, i) => <div key={i} className="warning">⚠ {w}</div>)}
      {plan.risks.map((r: Risk) => (
        <details key={r.id} className={'risk ' + r.level}>
          <summary>
            <span className={'lvl ' + r.level}>{r.level}</span>
            <span className="rtitle">{r.title}</span>
            <span className="mono rval">{fmt(r.value)} {r.unit}</span>
          </summary>
          <div className="rdetail">{r.detail}</div>
          <div className="rrule">Rule: {r.rule}</div>
        </details>
      ))}
      <div className="implant-metrics">
        Implant volume {fmt(plan.implant.volume_cc, 2)} cc · max thickness {fmt(plan.implant.max_thickness_mm)} mm ·
        bone contact {fmt(plan.implant.contact_area_cm2)} cm²
      </div>
    </div>
  )
}
