import { useCallback, useEffect, useMemo, useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim()
  || (import.meta.env.DEV ? 'http://localhost:8000/api' : '');
const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: 'grid' },
  { id: 'screening', label: 'New screening', icon: 'scan' },
  { id: 'queue', label: 'Review queue', icon: 'eye' },
  { id: 'reports', label: 'Reports', icon: 'report' },
];

const ICON_PATHS = {
  grid: 'M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z',
  scan: 'M4 7V4h3m10 0h3v3m0 10v3h-3M7 20H4v-3M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 2v4m-2-2h4',
  eye: 'M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7Zm10-3a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z',
  report: 'M7 3h8l4 4v14H5V3h2Zm7 1v4h4M8 12h8m-8 4h8',
  camera: 'M4 7h3l1.5-2h7L17 7h3v12H4V7Zm8 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z',
  heart: 'M20.8 8.6c0 5.2-8.8 10.2-8.8 10.2S3.2 13.8 3.2 8.6a4.3 4.3 0 0 1 8.8-.7 4.3 4.3 0 0 1 8.8.7ZM4 12h4l2-4 3 8 2-4h5',
  alert: 'M12 3 2.8 20h18.4L12 3Zm0 6v5m0 3h.01',
  check: 'm5 12 4 4L19 6',
  close: 'm6 6 12 12M18 6 6 18',
  menu: 'M4 6h16M4 12h16M4 18h16',
  arrow: 'M5 12h14m-6-6 6 6-6 6',
  moon: 'M20.5 14.5A8 8 0 0 1 9.5 3.5 8.5 8.5 0 1 0 20.5 14.5Z',
  sun: 'M12 3v2m0 14v2M5.6 5.6 7 7m10 10 1.4 1.4M3 12h2m14 0h2M5.6 18.4 7 17m10-10 1.4-1.4M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
  file: 'M14 2H6v20h12V8l-4-6Zm0 0v6h4M9 13h6m-6 4h6',
};

function Icon({ name, size = 18, className = '' }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={ICON_PATHS[name] || ICON_PATHS.file} />
    </svg>
  );
}

function isRecord(value) {
  return typeof value === 'object' && value !== null;
}

async function readResponse(response) {
  const body = await response.json();
  if (!response.ok) {
    const message = isRecord(body) && typeof body.error === 'string' ? body.error : '';
    throw new Error(message || `Request failed (${response.status}).`);
  }
  return body;
}

async function fetchScreenings(signal) {
  if (!API_BASE_URL) {
    throw new Error('Set VITE_API_BASE_URL to your deployed screening API URL.');
  }
  const response = await fetch(`${API_BASE_URL}/screenings`, { signal });
  const data = await readResponse(response);
  if (!Array.isArray(data.screenings)) {
    throw new Error('The screening service returned an invalid case list.');
  }
  return data.screenings;
}

function screeningStatus(screening) {
  return String(screening.status || 'PENDING').toUpperCase();
}

function formatDate(value) {
  if (!value) return 'Just now';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? 'Just now'
    : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

function StatCard({ label, value, hint, icon, tone }) {
  return (
    <article className={`stat-card stat-${tone}`}>
      <div className="stat-top">
        <span className="stat-icon"><Icon name={icon} size={19} /></span>
        <span className="stat-hint">{hint}</span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </article>
  );
}

function StatusBadge({ status }) {
  const label = status === 'COMPLETED'
    ? 'Complete'
    : status === 'PROCESSING'
      ? 'Processing'
      : status === 'FAILED'
        ? 'Failed'
        : 'Pending';
  return <span className={`status-badge status-${status.toLowerCase()}`}>{label}</span>;
}

function GradeBadge({ grade, referable }) {
  if (grade === null || grade === undefined) return <span className="muted-cell">Awaiting result</span>;
  const labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
  return (
    <span className={`grade-badge ${referable ? 'grade-refer' : 'grade-ok'}`}>
      ICDR {grade} · {labels[grade] || 'Unclassified'}
    </span>
  );
}

function CaseTable({ cases, emptyMessage, onOpen }) {
  if (!cases.length) {
    return (
      <div className="empty-state">
        <span className="empty-icon"><Icon name="camera" size={22} /></span>
        <strong>No screenings yet</strong>
        <span>{emptyMessage}</span>
      </div>
    );
  }
  return (
    <div className="table-scroll">
      <table className="case-table">
        <thead>
          <tr><th>Patient</th><th>Screening</th><th>Result</th><th>Status</th><th /></tr>
        </thead>
        <tbody>
          {cases.map((item) => {
            const status = screeningStatus(item);
            return (
              <tr key={item.screening_id || item.scan_id}>
                <td>
                  <strong>{item.patient_name || 'Unknown patient'}</strong>
                  <small>{item.patient_id || 'No patient ID'} · {formatDate(item.created_at)}</small>
                </td>
                <td><span className="case-id">#{item.screening_id || item.scan_id}</span></td>
                <td><GradeBadge grade={item.ai_grade} referable={item.is_referable} /></td>
                <td><StatusBadge status={status} /></td>
                <td className="table-action">
                  {item.fundus_image_url && (
                    <button className="text-action" onClick={() => onOpen(item)} type="button">
                      View <Icon name="arrow" size={15} />
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [screenings, setScreenings] = useState([]);
  const [loadingCases, setLoadingCases] = useState(true);
  const [apiError, setApiError] = useState('');
  const [selectedCase, setSelectedCase] = useState(null);

  const refreshScreenings = useCallback(async (signal) => {
    try {
      setScreenings(await fetchScreenings(signal));
      setApiError('');
    } catch (error) {
      if (error.name !== 'AbortError') {
        setApiError(error.message || 'Could not load screenings.');
      }
    } finally {
      if (!signal?.aborted) setLoadingCases(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetchScreenings(controller.signal)
      .then(setScreenings)
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setApiError(error.message || 'Could not load screenings.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingCases(false);
      });
    return () => controller.abort();
  }, []);

  const counts = useMemo(() => ({
    total: screenings.length,
    pending: screenings.filter((item) => ['PENDING', 'PROCESSING'].includes(screeningStatus(item))).length,
    referable: screenings.filter((item) => item.is_referable === true).length,
    failed: screenings.filter((item) => screeningStatus(item) === 'FAILED').length,
  }), [screenings]);

  function navigate(page) {
    setActivePage(page);
    setMobileNavOpen(false);
  }

  const pendingCases = screenings.filter((item) => ['PENDING', 'PROCESSING'].includes(screeningStatus(item)));
  const completedCases = screenings.filter((item) => screeningStatus(item) === 'COMPLETED');

  return (
    <div className={`app-frame ${darkMode ? 'dark-mode' : ''}`}>
      {mobileNavOpen && (
        <button
          className="mobile-scrim"
          onClick={() => setMobileNavOpen(false)}
          aria-label="Close navigation"
          type="button"
        />
      )}
      <aside className={`sidebar ${mobileNavOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <span className="brand-mark"><Icon name="eye" size={21} /></span>
          <span className="brand-name">Retina<span>Guard</span></span>
          <button className="mobile-close" onClick={() => setMobileNavOpen(false)} aria-label="Close menu" type="button">
            <Icon name="close" />
          </button>
        </div>
        <div className="brand-caption">DIABETIC RETINOPATHY SCREENING</div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => navigate(item.id)}
              className={`nav-link ${activePage === item.id ? 'nav-active' : ''}`}
            >
              <Icon name={item.icon} size={18} />
              <span>{item.label}</span>
              {item.id === 'queue' && pendingCases.length > 0 && (
                <span className="nav-count">{pendingCases.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="location-card">
            <span className="location-dot" />
            <span><strong>Screening workspace</strong><small>Clinical review desk</small></span>
          </div>
          <div className="sidebar-footnote">AI-assisted screening · ICDR 0–4</div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setMobileNavOpen(true)}
            aria-label="Open navigation"
            type="button"
          ><Icon name="menu" /></button>
          <div className="topbar-context">
            <span className="connection-indicator" />
            <span>Screening service</span>
            <span className="context-divider">/</span>
            <span className="context-current">{NAV_ITEMS.find((item) => item.id === activePage)?.label}</span>
          </div>
          <div className="topbar-actions">
            <button
              className="icon-button theme-button"
              onClick={() => setDarkMode((value) => !value)}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              type="button"
            ><Icon name={darkMode ? 'sun' : 'moon'} size={18} /></button>
            <button className="topbar-cta" onClick={() => navigate('screening')} type="button">
              <Icon name="scan" size={16} /> <span>New screening</span>
            </button>
          </div>
        </header>

        <main className="page-content">
          {apiError && (
            <div className="api-alert" role="status">
              <Icon name="alert" size={18} />
              <span>{apiError}</span>
              <button className="text-action" onClick={() => refreshScreenings()} type="button">Retry</button>
            </div>
          )}

          {activePage === 'dashboard' && (
            <section className="page-stack">
              <div className="welcome-banner">
                <div className="banner-orb banner-orb-one" />
                <div className="banner-orb banner-orb-two" />
                <div className="welcome-copy">
                  <div className="eyebrow"><span className="live-dot" /> RETINAL HEALTH WORKSPACE</div>
                  <h1>Screening desk</h1>
                  <p>Review retinal screenings and keep patients moving toward the care they need.</p>
                </div>
                <button className="banner-button" onClick={() => navigate('screening')} type="button">
                  <Icon name="scan" size={17} /> Start a screening <Icon name="arrow" size={16} />
                </button>
              </div>

              <div className="stats-grid">
                <StatCard label="Total screenings" value={loadingCases ? '—' : counts.total} hint="All records" icon="camera" tone="teal" />
                <StatCard label="Awaiting review" value={loadingCases ? '—' : counts.pending} hint="Human review" icon="heart" tone="blue" />
                <StatCard label="Referable results" value={loadingCases ? '—' : counts.referable} hint="ICDR level 2+" icon="alert" tone="coral" />
                <StatCard label="Needs attention" value={loadingCases ? '—' : counts.failed} hint="Pipeline errors" icon="scan" tone="amber" />
              </div>

              <div className="dashboard-grid">
                <section className="panel cases-panel">
                  <div className="panel-heading">
                    <div><h2>Recent screenings</h2><p>Latest patient images and their screening status.</p></div>
                    <button className="subtle-button" onClick={() => refreshScreenings()} type="button">Refresh</button>
                  </div>
                  {loadingCases ? <div className="loading-state">Loading screenings…</div> : (
                    <CaseTable
                      cases={screenings.slice(0, 7)}
                      emptyMessage="Start a screening to see patient cases here."
                      onOpen={setSelectedCase}
                    />
                  )}
                  {screenings.length > 7 && (
                    <button className="panel-footer-link" onClick={() => navigate('reports')} type="button">
                      View all screenings <Icon name="arrow" size={15} />
                    </button>
                  )}
                </section>
                <aside className="dashboard-aside">
                  <section className="panel workflow-panel">
                    <div className="panel-heading"><div><h2>Screening workflow</h2><p>From image to clinical review.</p></div></div>
                    <ol className="workflow-list">
                      <li><span className="workflow-number">01</span><span><strong>Capture</strong><small>Upload a clear fundus image.</small></span></li>
                      <li><span className="workflow-number">02</span><span><strong>Analyze</strong><small>Run the configured AI model.</small></span></li>
                      <li><span className="workflow-number">03</span><span><strong>Review</strong><small>Validate results with a clinician.</small></span></li>
                    </ol>
                    <button className="workflow-link" onClick={() => navigate('screening')} type="button">
                      Create screening <Icon name="arrow" size={16} />
                    </button>
                  </section>
                  <section className="guidance-card">
                    <span className="guidance-icon"><Icon name="heart" size={19} /></span>
                    <h3>Clinical review matters</h3>
                    <p>AI output supports screening decisions; it is not a diagnosis. Referable results should be reviewed by a qualified clinician.</p>
                  </section>
                </aside>
              </div>
            </section>
          )}

          {activePage === 'screening' && (
            <section className="page-stack narrow-page">
              <div className="page-heading">
                <div className="eyebrow">NEW PATIENT CASE</div>
                <h1>New screening</h1>
                <p>The patient intake form is being connected to this workspace.</p>
              </div>
              <section className="panel workflow-panel pending-form-panel">
                <div className="pending-form-icon"><Icon name="file" size={22} /></div>
                <h2>Patient form coming soon</h2>
                <p>The existing screening and patient forms will be connected here once the form components are added. No upload or patient record is submitted from this page yet.</p>
                <button className="subtle-button" onClick={() => navigate('dashboard')} type="button">Back to dashboard</button>
              </section>
            </section>
          )}

          {activePage === 'queue' && (
            <section className="page-stack">
              <div className="page-heading page-heading-row">
                <div><div className="eyebrow">CLINICAL WORKLIST</div><h1>Review queue</h1><p>Screenings still running or awaiting results.</p></div>
                <span className="queue-count">{pendingCases.length} open</span>
              </div>
              <section className="panel cases-panel">
                {loadingCases ? <div className="loading-state">Loading screenings…</div> : (
                  <CaseTable cases={pendingCases} emptyMessage="There are no pending screenings right now." onOpen={setSelectedCase} />
                )}
              </section>
            </section>
          )}

          {activePage === 'reports' && (
            <section className="page-stack">
              <div className="page-heading page-heading-row">
                <div><div className="eyebrow">CASE HISTORY</div><h1>Screening reports</h1><p>Completed results from the connected screening service.</p></div>
                <button className="subtle-button" onClick={() => refreshScreenings()} type="button">Refresh list</button>
              </div>
              <section className="panel cases-panel">
                {loadingCases ? <div className="loading-state">Loading screenings…</div> : (
                  <CaseTable cases={completedCases} emptyMessage="Completed screening reports will appear here." onOpen={setSelectedCase} />
                )}
              </section>
            </section>
          )}
        </main>
        <footer className="app-footer"><span>RetinaGuard</span><span>AI-assisted screening · Always seek qualified clinical review.</span></footer>
      </div>

      {selectedCase && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setSelectedCase(null);
        }}>
          <section className="case-modal" role="dialog" aria-modal="true" aria-labelledby="case-modal-title">
            <div className="modal-heading">
              <div><div className="eyebrow">SCREENING #{selectedCase.screening_id || selectedCase.scan_id}</div><h2 id="case-modal-title">{selectedCase.patient_name || 'Patient case'}</h2></div>
              <button className="icon-button" onClick={() => setSelectedCase(null)} aria-label="Close details" type="button"><Icon name="close" /></button>
            </div>
            <img className="modal-image" src={selectedCase.result_image_url || selectedCase.fundus_image_url} alt="Retinal screening image" />
            <div className="modal-details">
              <div><span>Patient ID</span><strong>{selectedCase.patient_id || 'Not provided'}</strong></div>
              <div><span>Age</span><strong>{selectedCase.age || 'Not provided'}</strong></div>
              <div><span>Status</span><StatusBadge status={screeningStatus(selectedCase)} /></div>
              <div><span>AI result</span><GradeBadge grade={selectedCase.ai_grade} referable={selectedCase.is_referable} /></div>
              {selectedCase.confidence !== null && selectedCase.confidence !== undefined && (
                <div><span>Model confidence</span><strong>{Math.round(selectedCase.confidence * 100)}%</strong></div>
              )}
            </div>
            <p className="modal-disclaimer">AI output is for screening support and does not constitute a medical diagnosis. A qualified clinician should review this result.</p>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
