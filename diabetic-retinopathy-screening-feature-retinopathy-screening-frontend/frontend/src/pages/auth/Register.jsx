import { Link } from "react-router-dom";
import "../../styles/auth.css";

export default function Register() {
  return (
    <div className="auth-page">

      <div className="auth-card">

        <div className="auth-header">
          <div className="auth-logo">◉</div>
          <h1>Create Professional Account</h1>
          <p>
            Request access to the DR-Screen AI platform
          </p>
        </div>

        <form className="auth-form">

          <div className="form-group">
            <label>Full Name</label>
            <input
              type="text"
              placeholder="Enter your full name"
            />
          </div>

          <div className="form-group">
            <label>Email Address</label>
            <input
              type="email"
              placeholder="Enter your email"
            />
          </div>

          <div className="form-group">
            <label>Phone Number</label>
            <input
              type="tel"
              placeholder="Enter your phone number"
            />
          </div>

          <div className="form-group">
            <label>Organization / Screening Center</label>
            <input
              type="text"
              placeholder="Enter organization name"
            />
          </div>

          <div className="form-group">
            <label>Center ID</label>
            <input
              type="text"
              placeholder="Enter center ID"
            />
          </div>

          <div className="form-row">

            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                placeholder="Create password"
              />
            </div>

            <div className="form-group">
              <label>Confirm Password</label>
              <input
                type="password"
                placeholder="Confirm password"
              />
            </div>

          </div>

          <label className="terms-checkbox">
            <input type="checkbox" />
            <span>
              I agree to the Terms of Use and Privacy Policy.
            </span>
          </label>

          <button type="submit" className="auth-button">
            Request Access →
          </button>

        </form>

        <div className="auth-footer">
          <span>Already have an account?</span>
          <Link to="/login">Sign in</Link>
        </div>

      </div>

    </div>
  );
}