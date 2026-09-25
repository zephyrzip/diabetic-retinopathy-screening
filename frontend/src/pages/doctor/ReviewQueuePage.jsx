import { Link } from "react-router-dom";
import "../../styles/doctor.css";
import "../../styles/operator.css";

export default function ReviewQueuePage() {
  const queueCases = [
    {
      id: "P-002",
      center: "St. Jude Clinic #4",
      date: "26 Sep 2026",
      aiSeverity: "Moderate DR (Stage 2)",
      confidence: "92.4%",
      referable: true,
      priority: "Urgent",
      findings: "Multiple microaneurysms, blot hemorrhages in superior quadrant",
    },
    {
      id: "P-003",
      center: "Apex Eye Foundation",
      date: "26 Sep 2026",
      aiSeverity: "Severe DR (Stage 3)",
      confidence: "97.1%",
      referable: true,
      priority: "Urgent",
      findings: "Venous beading, prominent cotton-wool spots, intraretinal microvascular abnormalities",
    },
    {
      id: "P-005",
      center: "Metro Rural Screening Unit",
      date: "26 Sep 2026",
      aiSeverity: "Proliferative DR (Stage 4)",
      confidence: "98.8%",
      referable: true,
      priority: "Critical",
      findings: "Neovascularization of the disc (NVD), preretinal fibrous proliferation",
    },
  ];

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
            <span className="nav-count-badge">3</span>
          </Link>

          <Link to="/doctor/review/P-002" className="doctor-nav-item">
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
                  <tr key={c.id}>
                    <td>
                      <strong>{c.id}</strong>
                    </td>
                    <td>{c.center}</td>
                    <td>{c.date}</td>
                    <td>
                      <span className="dr-level">{c.aiSeverity}</span>
                    </td>
                    <td>
                      <strong>{c.confidence}</strong>
                    </td>
                    <td>
                      <span style={{ fontSize: "11px", color: "#64748b" }}>
                        {c.findings}
                      </span>
                    </td>
                    <td>
                      <span
                        className={`priority-tag ${
                          c.priority === "Critical"
                            ? "priority-urgent"
                            : "priority-high"
                        }`}
                      >
                        {c.priority}
                      </span>
                    </td>
                    <td>
                      <Link
                        to={`/doctor/review/${c.id}`}
                        className="review-action-btn"
                      >
                        Evaluate & Certify →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
