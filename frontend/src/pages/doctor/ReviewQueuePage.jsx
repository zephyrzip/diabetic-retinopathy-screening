import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "../../styles/doctor.css";
import "../../styles/operator.css";
import { formatDate, gradeLabel, listScreenings } from "../../services/screenings";

export default function ReviewQueuePage() {
  const [queueCases, setQueueCases] = useState([]);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    listScreenings(controller.signal).then((data) => {
      setQueueCases((data.screenings || []).filter((screening) => screening.status === 'COMPLETED' && screening.is_referable && screening.review_status !== 'SIGNED'));
    }).catch((requestError) => {
      if (requestError.name !== 'AbortError') setError(requestError.message);
    });
    return () => controller.abort();
  }, []);

  return (
    <div className="doctor-layout">
      {/* SIDEBAR */}
      <aside className="doctor-sidebar">
        <div className="doctor-logo">
          <span>◉</span>
          <strong>DR-Screen AI</strong>
          <span className="doctor-portal-pill">DOCTOR</span>
        </div>

        <nav className="doctor-nav">
          <Link to="/doctor" className="doctor-nav-item">
            <span>⌂</span>
            <span>Dashboard</span>
          </Link>

          <Link to="/doctor/reviews" className="doctor-nav-item active">
            <span>📋</span>
            <span>Review Queue</span>
            <span className="nav-count-badge">{queueCases.length}</span>
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

      {/* MAIN */}
      <main className="doctor-main">
        <header className="doctor-topbar">
          <div>
            <p className="doctor-page-label">CLINICAL TRIAGE QUEUE</p>
            <h1>Ophthalmologist Review Queue</h1>
          </div>

          <div className="doctor-user">
            <Link to="/doctor" className="operator-primary-button" style={{ background: "#e0f4f4", color: "#087f8c" }}>
              ← Return to Dashboard
            </Link>
          </div>
        </header>

        <section className="doctor-panel">
          <div className="panel-header">
            <div>
              <span className="panel-label">ACTIVE CASES</span>
              <h2>Pending Clinical Evaluations ({queueCases.length})</h2>
            </div>
          </div>

          <div className="screening-table-wrapper">
            <table className="screening-table">
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Screening Center</th>
                  <th>Date</th>
                  <th>AI Predicted DR</th>
                  <th>Confidence</th>
                  <th>AI Lesion Findings</th>
                  <th>Urgency</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {queueCases.map((c) => (
                  <tr key={c.screening_id}>
                    <td>
                      <strong>{c.patient_id}</strong>
                    </td>
                    <td>{c.technician_id || 'Screening center'}</td>
                    <td>{formatDate(c.created_at)}</td>
                    <td>
                      <span className="dr-level">{gradeLabel(c.ai_grade)}</span>
                    </td>
                    <td>
                      <strong>{c.confidence == null ? '—' : `${Math.round(c.confidence * 100)}%`}</strong>
                    </td>
                    <td>
                      <span style={{ fontSize: "11px", color: "#64748b" }}>
                        AI report available for clinical review
                      </span>
                    </td>
                    <td>
                      <span
                        className={`priority-tag ${
                          c.ai_grade >= 4
                            ? "priority-urgent"
                            : "priority-high"
                        }`}
                      >
                        {c.ai_grade >= 4 ? 'Critical' : 'Urgent'}
                      </span>
                    </td>
                    <td>
                      <Link
                        to={`/doctor/review/${c.screening_id}`}
                        className="review-action-btn"
                      >
                        Evaluate & Certify →
                      </Link>
                    </td>
                  </tr>
                ))}
                {!queueCases.length && <tr><td colSpan="8">{error || 'No referable screenings are waiting for review.'}</td></tr>}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
