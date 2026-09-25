import { useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function LoginRoleModal({ isOpen, onClose }) {
  const navigate = useNavigate();

  // Close on ESC key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSelectRole = (role) => {
    onClose();
    navigate(`/login/${role}`);
  };

  return (
    <div className="login-modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className="login-modal-card"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          className="login-modal-close"
          onClick={onClose}
          aria-label="Close modal"
          type="button"
        >
          ✕
        </button>

        {/* Modal Header */}
        <div className="login-modal-header">
          <div className="login-modal-badge">
            <span>◉</span> PROFESSIONAL ACCESS
          </div>
          <h2>Select Login Profile</h2>
          <p>
            Choose your clinical or screening role to access the dedicated DR-Screen AI portal.
          </p>
        </div>

        {/* Two Login Options */}
        <div className="login-modal-grid">
          {/* Card 1: Ophthalmologist Profile */}
          <div
            className="role-selection-card doctor-role-card"
            onClick={() => handleSelectRole("ophthalmologist")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && handleSelectRole("ophthalmologist")}
          >
            <div className="role-card-badge doctor-badge">
              <span>SPECIALIST CLINICIAN</span>
            </div>

            <div className="role-card-icon-wrapper doctor-icon">
              <span className="role-glyph">🩺</span>
            </div>

            <h3>Ophthalmologist Profile</h3>
            <p className="role-card-desc">
              For retina specialists, eye clinics, and physicians reviewing AI classifications and validating diagnoses.
            </p>

            <ul className="role-card-highlights">
              <li>✓ Review referred diabetic retinopathy cases</li>
              <li>✓ Grad-CAM lesion explainability verification</li>
              <li>✓ Final clinical assessment & report sign-off</li>
            </ul>

            <button
              type="button"
              className="role-card-btn doctor-btn"
              onClick={(e) => {
                e.stopPropagation();
                handleSelectRole("ophthalmologist");
              }}
            >
              Sign In as Ophthalmologist →
            </button>

            <div className="role-card-footer">
              <span>New doctor?</span>{" "}
              <Link
                to="/register/doctor"
                onClick={onClose}
                className="role-register-link"
              >
                Create Account
              </Link>
            </div>
          </div>

          {/* Card 2: Screening Operator Profile */}
          <div
            className="role-selection-card operator-role-card"
            onClick={() => handleSelectRole("operator")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && handleSelectRole("operator")}
          >
            <div className="role-card-badge operator-badge">
              <span>SCREENING OPERATOR</span>
            </div>

            <div className="role-card-icon-wrapper operator-icon">
              <span className="role-glyph">🔬</span>
            </div>

            <h3>Screening Operator Profile</h3>
            <p className="role-card-desc">
              For technicians and primary healthcare operators capturing fundus images and generating instant AI screening results.
            </p>

            <ul className="role-card-highlights">
              <li>✓ Retinal fundus image acquisition & quality audit</li>
              <li>✓ Real-time 5-stage DR severity classification</li>
              <li>✓ Instant referable status & patient management</li>
            </ul>

            <button
              type="button"
              className="role-card-btn operator-btn"
              onClick={(e) => {
                e.stopPropagation();
                handleSelectRole("operator");
              }}
            >
              Sign In as Operator →
            </button>

            <div className="role-card-footer">
              <span>New operator?</span>{" "}
              <Link
                to="/register"
                onClick={onClose}
                className="role-register-link"
              >
                Create Account
              </Link>
            </div>
          </div>
        </div>

        {/* Modal Bottom Note */}
        <div className="login-modal-footer">
          <span>
            Need help selecting your portal? Contact clinical IT support or explore the{" "}
            <a href="#how-it-works" onClick={onClose}>
              system workflow
            </a>.
          </span>
        </div>
      </div>
    </div>
  );
}
