import { useState } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import "../../styles/doctor.css";
import "../../styles/operator.css";

export default function PatientReviewPage() {
  const { patientId = "P-002" } = useParams();
  const navigate = useNavigate();

  const [clinicalDecision, setClinicalDecision] = useState("Agree with AI (Moderate DR)");
  const [treatmentPlan, setTreatmentPlan] = useState("Fluorescein angiography and 3-month follow-up recommended.");
  const [signed, setSigned] = useState(false);

  const handleSignReport = (e) => {
    e.preventDefault();
    setSigned(true);
    setTimeout(() => {
      alert(`Report for patient ${patientId} successfully signed and transmitted to screening center.`);
      navigate("/doctor/reviews");
    }, 800);
  };

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

          <Link to="/doctor/reviews" className="doctor-nav-item">
            <span>📋</span>
            <span>Review Queue</span>
          </Link>

          <Link to={`/doctor/review/${patientId}`} className="doctor-nav-item active">
            <span>👁️</span>
            <span>Case Review</span>
          </Link>

          <Link to="/operator/explainability" className="doctor-nav-item">
            <span>✦</span>
            <span>Explainability</span>
          </Link>
        </nav>

        <div className="doctor-sidebar-bottom">
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
            <p className="doctor-page-label">CLINICAL VALIDATION & SIGN-OFF</p>
            <h1>Patient Case: {patientId}</h1>
          </div>

          <div className="doctor-user">
            <Link to="/doctor/reviews" className="operator-primary-button" style={{ background: "#e0f4f4", color: "#087f8c" }}>
              ← Back to Review Queue
            </Link>
          </div>
        </header>

        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "24px", marginBottom: "30px" }}>
          {/* AI Evidence Card */}
          <div className="doctor-panel" style={{ padding: "24px", margin: 0 }}>
            <span className="panel-label">IMAGE & AI GRAD-CAM EVIDENCE</span>
            <h3 style={{ margin: "8px 0 16px" }}>Fundus Image & Attention Heatmap</h3>

            <div style={{ background: "#10202f", borderRadius: "12px", padding: "16px", textAlign: "center", color: "white", marginBottom: "16px" }}>
              <div style={{ width: "100%", height: "260px", background: "radial-gradient(circle, #e65100 20%, #bf360c 60%, #1a0b00 100%)", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                <span style={{ background: "rgba(0,0,0,0.6)", padding: "6px 14px", borderRadius: "20px", fontSize: "12px" }}>
                  Fundus Macular Center • Grad-CAM Layer 4 Attention Overlay
                </span>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", background: "#f8fafc", padding: "14px", borderRadius: "10px" }}>
              <div>
                <small style={{ color: "#64748b" }}>AI DR Severity Prediction</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#ea580c" }}>Moderate DR (Stage 2)</p>
              </div>
              <div>
                <small style={{ color: "#64748b" }}>Model Confidence Score</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#087f8c" }}>92.4% (Concordant)</p>
              </div>
              <div>
                <small style={{ color: "#64748b" }}>Referral Recommendation</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#dc2626" }}>Referable to Specialist</p>
              </div>
              <div>
                <small style={{ color: "#64748b" }}>Quality Assessment</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#16a34a" }}>Adequate (Sharpness: 94%)</p>
              </div>
            </div>
          </div>

          {/* Doctor Assessment & Sign-Off Form */}
          <div className="doctor-panel" style={{ padding: "24px", margin: 0 }}>
            <span className="panel-label">OPHTHALMIC CLINICAL SIGN-OFF</span>
            <h3 style={{ margin: "8px 0 16px" }}>Physician Verification</h3>

            <form onSubmit={handleSignReport} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div className="form-group">
                <label style={{ fontSize: "13px", fontWeight: "600", color: "#334155" }}>
                  Clinical Finding Agreement
                </label>
                <select
                  value={clinicalDecision}
                  onChange={(e) => setClinicalDecision(e.target.value)}
                  style={{ width: "100%", padding: "11px", borderRadius: "8px", border: "1px solid #cbd5e1" }}
                >
                  <option value="Agree with AI (Moderate DR)">✓ Concur with AI: Moderate DR (Stage 2)</option>
                  <option value="Reclassified to Severe DR">Reclassify to Severe DR (Stage 3)</option>
                  <option value="Reclassified to Mild DR">Reclassify to Mild Non-Proliferative DR (Stage 1)</option>
                  <option value="Normal / No DR">Override: Normal / No Clinically Significant DR</option>
                </select>
              </div>

              <div className="form-group">
                <label style={{ fontSize: "13px", fontWeight: "600", color: "#334155" }}>
                  Specialist Recommendations & Clinical Plan
                </label>
                <textarea
                  rows={4}
                  value={treatmentPlan}
                  onChange={(e) => setTreatmentPlan(e.target.value)}
                  style={{ width: "100%", padding: "10px", borderRadius: "8px", border: "1px solid #cbd5e1", fontSize: "13px", boxSizing: "border-box" }}
                  placeholder="Enter medical directions..."
                  required
                />
              </div>

              <div style={{ background: "#f0fdfa", border: "1px solid #ccfbf1", borderRadius: "8px", padding: "12px", fontSize: "12px", color: "#115e59" }}>
                <strong>Digital Signature:</strong> Dr. Sarah Jenkins, MD (MCI-78291)
                <br />
                Timestamp: {new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}
              </div>

              <button
                type="submit"
                className="doctor-primary-button"
                style={{ width: "100%", padding: "14px", fontSize: "14px", border: "none", cursor: "pointer" }}
                disabled={signed}
              >
                {signed ? "Signing & Transmitting..." : "Sign & Certify Clinical Report →"}
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}
