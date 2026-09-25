import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "../../styles/auth.css";

export default function Register() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    phone: "",
    organization: "",
    centerId: "",
    password: "",
    confirmPassword: "",
    terms: false,
  });

  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (!formData.terms) {
      setError("Please agree to the Terms of Use and Privacy Policy.");
      return;
    }

    setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="auth-page">
        <div className="auth-card pending-card">
          <div className="pending-icon-wrapper">
            <div className="pending-badge-icon" style={{ background: "#e0f3f4", color: "#087f8c" }}>
              ✓
            </div>
          </div>

          <div className="auth-header">
            <span className="role-tag operator-tag">SCREENING OPERATOR ACCESS</span>
            <h1 style={{ marginTop: "12px" }}>Access Request Received</h1>
            <p>
              Your request for screening center <strong>{formData.organization || "Facility"}</strong> has been registered.
            </p>
          </div>

          <div className="pending-details-box">
            <div className="pending-step">
              <span className="step-icon done">✓</span>
              <div>
                <strong>Application Received</strong>
                <p>Operator {formData.fullName} ({formData.email}) registered.</p>
              </div>
            </div>

            <div className="pending-step">
              <span className="step-icon in-progress">◉</span>
              <div>
                <strong>Center ID Verification</strong>
                <p>Validating Center ID {formData.centerId || "Provided"} with regional health network.</p>
              </div>
            </div>
          </div>

          <div className="pending-actions">
            <button
              type="button"
              className="auth-button"
              onClick={() => navigate("/login/operator")}
            >
              Go to Operator Sign In →
            </button>
            <Link to="/" className="pending-secondary-link">
              ← Return to Home Page
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        {/* Top Nav */}
        <div className="auth-card-top-nav">
          <Link to="/" className="back-home-link">
            ← Back to Home
          </Link>
          <span className="brand-badge-pill">
            <span className="brand-dot">◉</span> DR-Screen AI
          </span>
        </div>

        {/* Role Banner / Cross-Link */}
        <div className="register-role-banner">
          <div className="register-role-banner-text">
            <span className="role-tag operator-tag">OPERATOR REGISTRATION</span>
            <p>Creating account for <strong>Screening Center Staff & Technicians</strong></p>
          </div>
          <Link to="/register/doctor" className="switch-register-link">
            Doctor? Register Here →
          </Link>
        </div>

        <div className="auth-header">
          <div className="auth-logo">◉</div>
          <h1>Create Screening Operator Account</h1>
          <p>
            Request access to the DR-Screen AI image acquisition & primary screening portal
          </p>
        </div>

        {error && <div className="auth-error-alert">{error}</div>}

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="reg-fullName">Full Name</label>
            <input
              id="reg-fullName"
              name="fullName"
              type="text"
              placeholder="e.g. Jane Doe"
              value={formData.fullName}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="reg-email">Email Address</label>
            <input
              id="reg-email"
              name="email"
              type="email"
              placeholder="operator@clinic.org"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="reg-phone">Phone Number</label>
            <input
              id="reg-phone"
              name="phone"
              type="tel"
              placeholder="+1 (555) 000-0000"
              value={formData.phone}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="reg-org">Organization / Screening Center</label>
            <input
              id="reg-org"
              name="organization"
              type="text"
              placeholder="St. Jude Eye Screening Center"
              value={formData.organization}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="reg-centerId">Center ID</label>
            <input
              id="reg-centerId"
              name="centerId"
              type="text"
              placeholder="CTR-8842"
              value={formData.centerId}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="reg-password">Password</label>
              <input
                id="reg-password"
                name="password"
                type="password"
                placeholder="Create password"
                value={formData.password}
                onChange={handleChange}
                required
                minLength={6}
              />
            </div>

            <div className="form-group">
              <label htmlFor="reg-confirmPassword">Confirm Password</label>
              <input
                id="reg-confirmPassword"
                name="confirmPassword"
                type="password"
                placeholder="Confirm password"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          <label className="terms-checkbox">
            <input
              type="checkbox"
              name="terms"
              checked={formData.terms}
              onChange={handleChange}
            />
            <span>
              I agree to the Terms of Use, Clinical Privacy Policy & Data Security Standards.
            </span>
          </label>

          <button type="submit" className="auth-button">
            Request Center Access →
          </button>
        </form>

        <div className="auth-footer">
          <span>Already have an account?</span>
          <Link to="/login/operator">Sign in as Operator</Link>
        </div>
      </div>
    </div>
  );
}