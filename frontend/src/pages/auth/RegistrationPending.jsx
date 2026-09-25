import { Link } from "react-router-dom";
import "../../styles/auth.css";

export default function RegistrationPending() {
  return (
    <div className="auth-page">
      <div className="auth-card pending-card">
        <div className="pending-icon-wrapper">
          <div className="pending-badge-icon">⏳</div>
        </div>

        <div className="auth-header">
          <span className="role-tag doctor-tag">OPHTHALMOLOGIST APPLICATION</span>
          <h1 style={{ marginTop: "12px" }}>Registration Submitted</h1>
          <p>
            Your clinical account application has been received and is currently under verification.
          </p>
        </div>

        <div className="pending-details-box">
          <div className="pending-step">
            <span className="step-icon done">✓</span>
            <div>
              <strong>Profile & Credentials Received</strong>
              <p>Medical registration and hospital affiliation details saved.</p>
            </div>
          </div>

          <div className="pending-step">
            <span className="step-icon in-progress">◉</span>
            <div>
              <strong>Medical License Verification</strong>
              <p>Our clinical compliance team verifies credentials with the Medical Council (typically 12–24 hours).</p>
            </div>
          </div>

          <div className="pending-step">
            <span className="step-icon pending">○</span>
            <div>
              <strong>Portal Access Activation</strong>
              <p>You will receive an activation email with your secure clinical login credentials.</p>
            </div>
          </div>
        </div>

        <div className="pending-actions">
          <Link to="/login/ophthalmologist" className="auth-button">
            Go to Ophthalmologist Login →
          </Link>
          <Link to="/" className="pending-secondary-link">
            ← Return to Home Page
          </Link>
        </div>
      </div>
    </div>
  );
}
