import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { formatDate, getScreening, gradeLabel } from '../../services/screenings';
import '../../styles/operator.css';

export default function ScreeningResult() {
  const { screeningId } = useParams();
  const [screening, setScreening] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!screeningId) return undefined;
    let cancelled = false;
    const load = async () => {
      try {
        const next = await getScreening(screeningId);
        if (!cancelled) { setScreening(next); setError(''); }
      } catch (requestError) {
        if (!cancelled) setError(requestError.message);
      }
    };
    load();
    const interval = window.setInterval(load, 3000);
    return () => { cancelled = true; window.clearInterval(interval); };
  }, [screeningId]);

  const status = screening?.status || 'LOADING';
  return (
    <div className="operator-layout">
      <aside className="operator-sidebar"><div className="operator-logo"><span>◉</span><strong>DR-Screen AI</strong></div><nav className="operator-nav"><Link to="/operator" className="operator-nav-item"><span>⌂</span>Dashboard</Link><Link to="/operator/new-screening" className="operator-nav-item"><span>＋</span>New Screening</Link></nav></aside>
      <main className="operator-main">
        <header className="operator-topbar"><div><p className="operator-page-label">AI SCREENING</p><h1>Screening #{screeningId}</h1></div><Link to="/operator" className="operator-primary-button">Back to dashboard</Link></header>
        {error && <p role="alert" style={{ color: '#b91c1c' }}>{error}</p>}
        {!screening && !error && <p>Loading screening…</p>}
        {screening && <section className="operator-panel" style={{ maxWidth: 900 }}>
          <div className="panel-header"><div><span className="panel-label">{status}</span><h2>{screening.patient_name}</h2></div></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(280px, 1fr)', gap: 24 }}>
            {screening.result_image_url || screening.fundus_image_url ? <img src={screening.result_image_url || screening.fundus_image_url} alt="Uploaded retinal fundus" style={{ width: '100%', borderRadius: 12, maxHeight: 430, objectFit: 'contain', background: '#10202f' }} /> : <div />}
            <div style={{ display: 'grid', alignContent: 'start', gap: 12 }}>
              <p><strong>Patient ID:</strong> {screening.patient_id}</p><p><strong>Created:</strong> {formatDate(screening.created_at)}</p><p><strong>AI result:</strong> {status === 'COMPLETED' ? gradeLabel(screening.ai_grade) : 'Awaiting processing'}</p><p><strong>Confidence:</strong> {screening.confidence == null ? '—' : `${Math.round(screening.confidence * 100)}%`}</p><p><strong>Recommendation:</strong> {screening.is_referable == null ? 'Awaiting processing' : screening.is_referable ? 'Refer for clinical review' : 'Non-referable'}</p>
              {status === 'PROCESSING' && <p>Processing is running. This page refreshes automatically.</p>}
              {status === 'FAILED' && <p role="alert" style={{ color: '#b91c1c' }}>{screening.processing_error || 'Processing failed.'}</p>}
              {status === 'COMPLETED' && screening.is_referable && <Link to="/doctor/reviews" className="operator-primary-button">Open review queue</Link>}
            </div>
          </div>
        </section>}
      </main>
    </div>
  );
}
