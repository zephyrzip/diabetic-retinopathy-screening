import { useEffect, useState } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import "../../styles/doctor.css";
import "../../styles/operator.css";
import { getScreening, gradeLabel, signScreeningReview } from '../../services/screenings';

export default function PatientReviewPage() {
  const { screeningId } = useParams();
  const navigate = useNavigate();

  const [screening, setScreening] = useState(null);
  const [clinicalDecision, setClinicalDecision] = useState('Agree with AI assessment');
  const [treatmentPlan, setTreatmentPlan] = useState('');
  const [doctorId, setDoctorId] = useState('');
  const [signed, setSigned] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    getScreening(screeningId, controller.signal).then(setScreening).catch((requestError) => {
      if (requestError.name !== 'AbortError') setError(requestError.message);
    });
    return () => controller.abort();
  }, [screeningId]);

  const handleSignReport = async (e) => {
    e.preventDefault();
    setError('');
    setSigned(true);
    try {
      await signScreeningReview(screeningId, { clinical_decision: clinicalDecision, doctor_notes: treatmentPlan, doctor_id: doctorId });
      navigate("/doctor/reviews");
    } catch (requestError) {
      setError(requestError.message);
      setSigned(false);
    }
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

          <Link to={`/doctor/review/${screeningId}`} className="doctor-nav-item active">
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
            <h1>Patient Case: {screening?.patient_id || `#${screeningId}`}</h1>
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
              {screening?.result_image_url || screening?.fundus_image_url ? <img src={screening.result_image_url || screening.fundus_image_url} alt="Retinal fundus submitted for review" style={{ width: '100%', height: 260, objectFit: 'contain', borderRadius: 10 }} /> : <span>Loading retinal image…</span>}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", background: "#f8fafc", padding: "14px", borderRadius: "10px" }}>
              <div>
                <small style={{ color: "#64748b" }}>AI DR Severity Prediction</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#ea580c" }}>{screening ? gradeLabel(screening.ai_grade) : '—'}</p>
              </div>
              <div>
                <small style={{ color: "#64748b" }}>Model Confidence Score</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#087f8c" }}>{screening?.confidence == null ? '—' : `${Math.round(screening.confidence * 100)}%`}</p>
              </div>
              <div>
                <small style={{ color: "#64748b" }}>Referral Recommendation</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#dc2626" }}>{screening?.is_referable ? 'Referable to specialist' : 'Non-referable'}</p>
              </div>
              <div>
                <small style={{ color: "#64748b" }}>Quality Assessment</small>
                <p style={{ margin: "2px 0 0", fontWeight: "700", color: "#16a34a" }}>{screening?.status || 'Loading'}</p>
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
                  Doctor ID
                </label>
                <input value={doctorId} onChange={(e) => setDoctorId(e.target.value)} style={{ width: "100%", padding: "11px", borderRadius: "8px", border: "1px solid #cbd5e1", boxSizing: "border-box" }} required />
              </div>

              <div className="form-group">
                <label style={{ fontSize: "13px", fontWeight: "600", color: "#334155" }}>
                  Clinical Finding Agreement
                </label>
                <select
                  value={clinicalDecision}
                  onChange={(e) => setClinicalDecision(e.target.value)}
                  style={{ width: "100%", padding: "11px", borderRadius: "8px", border: "1px solid #cbd5e1" }}
                >
                  <option value="Agree with AI assessment">✓ Concur with AI assessment</option>
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

              <div style={{ background: "#f0fdfa", border: "1px solid #ccfbf1", borderRadius: "8px", padding: "12px", fontSize: "12px", color: "#115e59" }}><strong>Digital signature:</strong> The signed review is stored with the doctor ID entered above.</div>
              {error && <p role="alert" style={{ color: '#b91c1c', margin: 0 }}>{error}</p>}

              <button
                type="submit"
                className="doctor-primary-button"
                style={{ width: "100%", padding: "14px", fontSize: "14px", border: "none", cursor: "pointer" }}
                disabled={signed || !screening || screening.status !== 'COMPLETED'}
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
