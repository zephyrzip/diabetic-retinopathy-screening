import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import "../../styles/doctor.css";
import "../../styles/operator.css";
import { formatDate, gradeLabel, listScreenings } from '../../services/screenings';

export default function DoctorDashboard() {
  const [screenings, setScreenings] = useState([]);
  useEffect(() => {
    const controller = new AbortController();
    listScreenings(controller.signal).then((data) => setScreenings(data.screenings || [])).catch(() => {});
    return () => controller.abort();
  }, []);
  const pendingReviews = useMemo(() => screenings.filter((screening) => screening.status === 'COMPLETED' && screening.is_referable && screening.review_status !== 'SIGNED').map((screening) => {
    const [date = '—', time = ''] = formatDate(screening.created_at).split(',');
    return { id: screening.screening_id, date, time: time.trim(), level: gradeLabel(screening.ai_grade), confidence: screening.confidence == null ? '—' : `${Math.round(screening.confidence * 100)}%`, priority: screening.ai_grade >= 4 ? 'Critical' : 'Urgent', lesions: 'AI report ready for clinical review' };
  }), [screenings]);

  return (
    <div className="doctor-layout">
      {/* =========================
          DOCTOR SIDEBAR
      ========================= */}
      <aside className="doctor-sidebar">
        <div className="doctor-logo">
          <span>◉</span>
          <strong>DR-Screen AI</strong>
          <span className="doctor-portal-pill">DOCTOR</span>
        </div>

        <nav className="doctor-nav">
          <Link to="/doctor" className="doctor-nav-item active">
            <span>⌂</span>
            <span>Dashboard</span>
          </Link>

          <Link to="/doctor/reviews" className="doctor-nav-item">
            <span>📋</span>
            <span>Review Queue</span>
            <span className="nav-count-badge">3</span>
          </Link>

          <Link to="/doctor/reviews" className="doctor-nav-item">
            <span>👁️</span>
            <span>Case Review</span>
          </Link>

          <Link to="/operator/explainability" className="doctor-nav-item">
            <span>✦</span>
            <span>Explainability</span>
          </Link>
        </nav>

        <div className="doctor-sidebar-bottom">
          <Link to="/doctor" className="doctor-nav-item">
            <span>⚙</span>
            <span>Clinical Settings</span>
          </Link>

          <Link to="/login/ophthalmologist" className="doctor-nav-item logout">
            <span>↪</span>
            <span>Sign Out</span>
          </Link>
        </div>
      </aside>

      {/* =========================
          MAIN CLINICAL AREA
      ========================= */}
      <main className="doctor-main">
        {/* TOPBAR */}
        <header className="doctor-topbar">
          <div>
            <p className="doctor-page-label">OPHTHALMIC CLINICAL SUITE</p>
            <h1>Doctor Review Dashboard</h1>
          </div>

          <div className="doctor-user">
            <div className="notification">
              🔔
              <span></span>
            </div>

            <div className="doctor-avatar">DJ</div>

            <div className="doctor-user-info">
              <strong>Dr. Sarah Jenkins, MD</strong>
              <small>Vitreo-Retinal Specialist • Lic #MCI-78291</small>
            </div>
          </div>
        </header>

        {/* WELCOME BANNER */}
        <section className="doctor-welcome">
          <div>
            <span className="welcome-label">CLINICAL TRIAGE STATUS</span>
            <h2>Welcome back, Dr. Jenkins</h2>
            <p>
              You have 3 referred diabetic retinopathy cases flagged by the AI engine awaiting your clinical validation and Grad-CAM review.
            </p>
          </div>

          <Link to="/doctor/reviews" className="doctor-primary-button">
            Open Review Queue ({pendingReviews.length}) →
          </Link>
        </section>

        {/* CLINICAL STATS */}
        <section className="doctor-stats">
          <div className="doctor-stat-card">
            <div className="stat-icon referable">!</div>
            <div>
              <span>Pending Reviews</span>
              <strong>{pendingReviews.length}</strong>
              <small>Requires clinical sign-off</small>
            </div>
          </div>

          <div className="doctor-stat-card">
            <div className="stat-icon" style={{ background: "#fff7ed", color: "#ea580c" }}>
              ⚡
            </div>
            <div>
              <span>Urgent / PDR</span>
              <strong>{pendingReviews.filter((patient) => patient.priority === 'Critical').length}</strong>
              <small>Critical referral cases</small>
            </div>
          </div>

          <div className="doctor-stat-card">
            <div className="stat-icon quality">✓</div>
            <div>
              <span>Validated Today</span>
              <strong>{screenings.filter((screening) => screening.review_status === 'SIGNED').length}</strong>
              <small>Reports certified</small>
            </div>
          </div>

          <div className="doctor-stat-card">
            <div className="stat-icon" style={{ background: "#e0f4f4", color: "#087f8c" }}>
              ◉
            </div>
            <div>
              <span>AI Concordance</span>
              <strong>—</strong>
              <small>Available after review</small>
            </div>
          </div>
        </section>

        {/* PATIENT CASES TABLE */}
        <section className="doctor-panel">
          <div className="panel-header">
            <div>
              <span className="panel-label">ACTIVE CLINICAL QUEUE</span>
              <h2>Referred Patients Requiring Doctor Review</h2>
            </div>

            <Link to="/doctor/reviews">View All Cases →</Link>
          </div>

          <div className="screening-table-wrapper">
            <table className="screening-table">
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Screening Time</th>
                  <th>AI Predicted DR</th>
                  <th>Confidence</th>
                  <th>Key Lesion Findings</th>
                  <th>Priority</th>
                  <th>Action</th>
                </tr>
              </thead>

              <tbody>
                {pendingReviews.map((patient) => (
                  <tr key={patient.id}>
                    <td>
                      <strong>{patient.id}</strong>
                    </td>
                    <td>
                      {patient.date} <small style={{ color: "#94a3b8" }}>{patient.time}</small>
                    </td>
                    <td>
                      <span className="dr-level">{patient.level}</span>
                    </td>
                    <td>
                      <strong>{patient.confidence}</strong>
                    </td>
                    <td>
                      <span style={{ fontSize: "11px", color: "#64748b" }}>
                        {patient.lesions}
                      </span>
                    </td>
                    <td>
                      <span
                        className={`priority-tag ${
                          patient.priority === "Critical"
                            ? "priority-urgent"
                            : patient.priority === "Urgent"
                            ? "priority-high"
                            : "priority-normal"
                        }`}
                      >
                        {patient.priority}
                      </span>
                    </td>
                    <td>
                      <Link
                        to={`/doctor/review/${patient.id}`}
                        className="review-action-btn"
                      >
                        Review Case →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* QUICK CLINICAL ACTIONS */}
        <section className="quick-actions">
          <h2>Specialist Actions</h2>

          <div className="quick-action-grid">
            <Link to="/doctor/reviews" className="quick-action-card">
              <span>📋</span>
              <div>
                <strong>Examine Full Queue</strong>
                <p>Filter by DR severity, center ID, or urgency</p>
              </div>
              <b>→</b>
            </Link>

            <Link to="/operator/explainability" className="quick-action-card">
              <span>✦</span>
              <div>
                <strong>Grad-CAM Inspector</strong>
                <p>Verify neural network attention heatmaps</p>
              </div>
              <b>→</b>
            </Link>

            <Link to="/doctor/reviews" className="quick-action-card">
              <span>📑</span>
              <div>
                <strong>Digital Clinical Report</strong>
                <p>Sign off on clinical recommendations</p>
              </div>
              <b>→</b>
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
