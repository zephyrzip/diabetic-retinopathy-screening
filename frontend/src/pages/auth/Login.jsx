import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import "../../styles/auth.css";

export default function Login({ initialRole: propRole }) {
  const navigate = useNavigate();
  const location = useLocation();

  // Determine base role from props, path, or query string
  const getBaseRole = () => {
    if (propRole) return propRole;

    const path = location.pathname.toLowerCase();
    if (path.includes("doctor") || path.includes("ophthalmologist")) {
      return "ophthalmologist";
    }
    if (path.includes("operator")) {
      return "operator";
    }

    const searchParams = new URLSearchParams(location.search);
    const roleParam = searchParams.get("role");
    if (roleParam === "doctor" || roleParam === "ophthalmologist") {
      return "ophthalmologist";
    }
    if (roleParam === "operator") {
      return "operator";
    }

    return "ophthalmologist"; // default
  };

  const baseRole = getBaseRole();
  const [manualRole, setManualRole] = useState(null);
  const [prevPath, setPrevPath] = useState(location.pathname);

  if (prevPath !== location.pathname) {
    setPrevPath(location.pathname);
    setManualRole(null);
  }

  const role = manualRole || baseRole;
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // Doctor Form State
  const [doctorForm, setDoctorForm] = useState({
    identifier: "",
    password: "",
  });

  // Operator Form State
  const [operatorForm, setOperatorForm] = useState({
    identifier: "",
    password: "",
  });

  // Handle Quick Demo Fill
  const handleDemoFill = () => {
    setErrorMsg("");
    if (role === "ophthalmologist") {
      setDoctorForm({
        identifier: "dr.jenkins@retinascreen.org",
        password: "DoctorPass2026!",
      });
    } else {
      setOperatorForm({
        identifier: "operator.alex@retinascreen.org",
        password: "OperatorPass2026!",
      });
    }
  };

  const handleRoleSwitch = (newRole) => {
    setManualRole(newRole);
    setErrorMsg("");
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setErrorMsg("");

    const currentForm = role === "ophthalmologist" ? doctorForm : operatorForm;

    if (!currentForm.identifier.trim() || !currentForm.password.trim()) {
      setErrorMsg("Please fill in both your identifier and password.");
      return;
    }

    setIsLoading(true);

    // Simulate login delay & storage
    setTimeout(() => {
      setIsLoading(false);
      if (role === "ophthalmologist") {
        sessionStorage.setItem(
          "user_session",
          JSON.stringify({
            role: "doctor",
            title: "Ophthalmologist",
            name: "Dr. Sarah Jenkins, MD",
            identifier: doctorForm.identifier,
          })
        );
        navigate("/doctor");
      } else {
        sessionStorage.setItem(
          "user_session",
          JSON.stringify({
            role: "operator",
            title: "Screening Operator",
            name: "Alex Rivera",
            identifier: operatorForm.identifier,
          })
        );
        navigate("/operator");
      }
    }, 500);
  };

  const isDoctor = role === "ophthalmologist";

  return (
    <div className="auth-page">
      <div className="auth-card login-card-enhanced">
        {/* Top Navigation */}
        <div className="auth-card-top-nav">
          <Link to="/" className="back-home-link">
            ← Back to Home
          </Link>

          <span className="brand-badge-pill">
            <span className="brand-dot">◉</span> DR-Screen AI
          </span>
        </div>

        {/* Profile Switcher Tabs */}
        <div className="role-tabs-container">
          <button
            type="button"
            className={`role-tab-btn ${isDoctor ? "active doctor-active" : ""}`}
            onClick={() => handleRoleSwitch("ophthalmologist")}
          >
            <span className="role-tab-icon">🩺</span>
            <div className="role-tab-text">
              <strong>Ophthalmologist Profile</strong>
              <small>Clinical Review & Validation</small>
            </div>
          </button>

          <button
            type="button"
            className={`role-tab-btn ${!isDoctor ? "active operator-active" : ""}`}
            onClick={() => handleRoleSwitch("operator")}
          >
            <span className="role-tab-icon">🔬</span>
            <div className="role-tab-text">
              <strong>Screening Operator Profile</strong>
              <small>Fundus Image & Quality Audit</small>
            </div>
          </button>
        </div>

        {/* Header with Role Details */}
        <div className="auth-header" style={{ marginTop: "18px", marginBottom: "20px" }}>
          <div className={`auth-profile-badge ${isDoctor ? "doctor-portal-badge" : "operator-portal-badge"}`}>
            <span className="profile-badge-icon">{isDoctor ? "🩺" : "🔬"}</span>
            <span>{isDoctor ? "OPHTHALMOLOGIST PORTAL" : "SCREENING OPERATOR PORTAL"}</span>
          </div>

          <h1>{isDoctor ? "Ophthalmologist Login" : "Screening Operator Login"}</h1>
          <p>
            {isDoctor
              ? "Sign in to review AI screening predictions, Grad-CAM explainability, and validate clinical cases."
              : "Sign in to acquire retinal images, verify quality metrics, and generate instant AI DR reports."}
          </p>
        </div>

        {/* Quick Demo Credentials Bar */}
        <div className="demo-credentials-banner">
          <div className="demo-info">
            <span className="demo-pill">TEST DEMO</span>
            <span className="demo-text">
              {isDoctor
                ? "Dr. Jenkins (dr.jenkins@retinascreen.org)"
                : "Operator Alex (operator.alex@retinascreen.org)"}
            </span>
          </div>
          <button
            type="button"
            onClick={handleDemoFill}
            className="demo-autofill-btn"
          >
            Autofill Demo Login
          </button>
        </div>

        {/* Error Alert */}
        {errorMsg && <div className="auth-error-alert">{errorMsg}</div>}

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {/* Identifier Field */}
          <div className="form-group">
            <label htmlFor="login-identifier">
              {isDoctor
                ? "Doctor Email or Medical Reg. Number"
                : "Operator Email or Screening Center ID"}
            </label>
            <input
              id="login-identifier"
              type="text"
              placeholder={
                isDoctor
                  ? "e.g. dr.jenkins@retinascreen.org or MCI-78291"
                  : "e.g. operator.alex@retinascreen.org or CTR-104"
              }
              value={isDoctor ? doctorForm.identifier : operatorForm.identifier}
              onChange={(e) => {
                const val = e.target.value;
                if (isDoctor) {
                  setDoctorForm((prev) => ({ ...prev, identifier: val }));
                } else {
                  setOperatorForm((prev) => ({ ...prev, identifier: val }));
                }
              }}
              required
            />
          </div>

          {/* Password Field */}
          <div className="form-group">
            <div className="form-label-row">
              <label htmlFor="login-password">Password</label>
              <a
                href="#forgot-password"
                className="forgot-link"
                onClick={(e) => {
                  e.preventDefault();
                  alert("Password reset instructions have been sent to your administrator.");
                }}
              >
                Forgot Password?
              </a>
            </div>

            <div className="password-field">
              <input
                id="login-password"
                type={showPassword ? "text" : "password"}
                placeholder="Enter your secure password"
                value={isDoctor ? doctorForm.password : operatorForm.password}
                onChange={(e) => {
                  const val = e.target.value;
                  if (isDoctor) {
                    setDoctorForm((prev) => ({ ...prev, password: val }));
                  } else {
                    setOperatorForm((prev) => ({ ...prev, password: val }));
                  }
                }}
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="password-toggle"
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </div>

          {/* Remember Me Checkbox */}
          <div className="login-options-row">
            <label className="terms-checkbox">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <span>Remember this professional workstation</span>
            </label>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="auth-button login-submit-btn"
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="loading-spinner-row">
                <span className="spinner"></span> Signing In...
              </span>
            ) : isDoctor ? (
              "Sign In as Ophthalmologist →"
            ) : (
              "Sign In as Screening Operator →"
            )}
          </button>
        </form>

        {/* Create an Account Option */}
        <div className="create-account-card">
          <div className="create-account-text">
            <span>Don't have an account?</span>
            <strong>
              {isDoctor
                ? "Register as an Ophthalmologist"
                : "Create a Screening Operator Account"}
            </strong>
          </div>
          <Link
            to={isDoctor ? "/register/doctor" : "/register"}
            className="create-account-btn"
          >
            Create an Account →
          </Link>
        </div>

        {/* Alternate Profile Switch Helper */}
        <div className="auth-footer alternate-switch">
          <span>
            {isDoctor
              ? "Looking for primary image screening?"
              : "Looking for clinical validation queue?"}
          </span>
          <button
            type="button"
            className="link-switch-button"
            onClick={() =>
              handleRoleSwitch(isDoctor ? "operator" : "ophthalmologist")
            }
          >
            {isDoctor
              ? "Switch to Screening Operator Profile"
              : "Switch to Ophthalmologist Profile"}
          </button>
        </div>
      </div>
    </div>
  );
}
