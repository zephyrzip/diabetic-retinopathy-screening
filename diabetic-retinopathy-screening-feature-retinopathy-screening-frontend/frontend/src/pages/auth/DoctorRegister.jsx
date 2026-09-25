import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function DoctorRegister() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    phone: "",
    registrationNumber: "",
    hospital: "",
    specialization: "Ophthalmology",
    password: "",
    confirmPassword: "",
    terms: false,
  });

  const [certificate, setCertificate] = useState(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      alert("Passwords do not match.");
      return;
    }

    if (!formData.terms) {
      alert("Please agree to the Terms & Privacy Policy.");
      return;
    }

    console.log("Doctor registration:", {
      ...formData,
      certificate,
    });

    // Prototype behaviour
    navigate("/registration-pending");
  };

  return (
    <div className="register-page">

      <div className="register-card">

        {/* Header */}
        <div className="register-header">

          <Link to="/" className="register-logo">
            <span className="register-logo-icon">◉</span>
            <span>DR-Screen AI</span>
          </Link>

          <h1>Create Ophthalmologist Account</h1>

          <p>
            Register to review AI-assisted retinal screenings.
          </p>

        </div>


        <form onSubmit={handleSubmit}>

          {/* Personal Information */}
          <div className="form-section">

            <div className="form-section-title">
              Personal Information
            </div>

            <div className="form-group">

              <label htmlFor="fullName">
                Full Name
              </label>

              <input
                id="fullName"
                name="fullName"
                type="text"
                placeholder="Dr. John Doe"
                value={formData.fullName}
                onChange={handleChange}
                required
              />

            </div>


            <div className="form-row">

              <div className="form-group">

                <label htmlFor="email">
                  Email Address
                </label>

                <input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="doctor@example.com"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />

              </div>


              <div className="form-group">

                <label htmlFor="phone">
                  Phone Number
                </label>

                <input
                  id="phone"
                  name="phone"
                  type="tel"
                  placeholder="+91 XXXXX XXXXX"
                  value={formData.phone}
                  onChange={handleChange}
                  required
                />

              </div>

            </div>

          </div>


          {/* Professional Information */}
          <div className="form-section">

            <div className="form-section-title">
              Professional Information
            </div>


            <div className="form-group">

              <label htmlFor="registrationNumber">
                Medical Registration Number
              </label>

              <input
                id="registrationNumber"
                name="registrationNumber"
                type="text"
                placeholder="Enter registration number"
                value={formData.registrationNumber}
                onChange={handleChange}
                required
              />

            </div>


            <div className="form-group">

              <label htmlFor="hospital">
                Hospital / Clinic
              </label>

              <input
                id="hospital"
                name="hospital"
                type="text"
                placeholder="Hospital or clinic name"
                value={formData.hospital}
                onChange={handleChange}
                required
              />

            </div>


            <div className="form-group">

              <label htmlFor="specialization">
                Specialization
              </label>

              <select
                id="specialization"
                name="specialization"
                value={formData.specialization}
                onChange={handleChange}
              >
                <option value="Ophthalmology">
                  Ophthalmology
                </option>

                <option value="Vitreo-Retinal Specialist">
                  Vitreo-Retinal Specialist
                </option>

                <option value="Other">
                  Other
                </option>

              </select>

            </div>

          </div>


          {/* Verification */}
          <div className="form-section">

            <div className="form-section-title">
              Professional Verification
            </div>

            <div className="form-group">

              <label htmlFor="certificate">
                Registration Certificate
              </label>

              <div className="file-upload">

                <input
                  id="certificate"
                  name="certificate"
                  type="file"
                  accept=".pdf,.jpg,.jpeg,.png"
                  onChange={(e) =>
                    setCertificate(e.target.files[0])
                  }
                />

                <span>
                  {certificate
                    ? certificate.name
                    : "Choose verification document"}
                </span>

              </div>

              <small>
                PDF, JPG or PNG
              </small>

            </div>

          </div>


          {/* Security */}
          <div className="form-section">

            <div className="form-section-title">
              Account Security
            </div>


            <div className="form-group">

              <label htmlFor="password">
                Password
              </label>

              <div className="password-field">

                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Create a strong password"
                  value={formData.password}
                  onChange={handleChange}
                  required
                  minLength={8}
                />

                <button
                  type="button"
                  onClick={() =>
                    setShowPassword(!showPassword)
                  }
                  className="password-toggle"
                >
                  {showPassword ? "Hide" : "Show"}
                </button>

              </div>

            </div>


            <div className="form-group">

              <label htmlFor="confirmPassword">
                Confirm Password
              </label>

              <div className="password-field">

                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={
                    showConfirmPassword
                      ? "text"
                      : "password"
                  }
                  placeholder="Confirm your password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  required
                />

                <button
                  type="button"
                  onClick={() =>
                    setShowConfirmPassword(
                      !showConfirmPassword
                    )
                  }
                  className="password-toggle"
                >
                  {showConfirmPassword ? "Hide" : "Show"}
                </button>

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
                I agree to the Terms & Privacy Policy
              </span>

            </label>

          </div>


          {/* Submit */}
          <button
            type="submit"
            className="register-button"
          >
            Submit Registration →
          </button>


          {/* Login */}
          <p className="login-link">

            Already have an account?

            <Link to="/login">
              Login
            </Link>

          </p>

        </form>

      </div>

    </div>
  );
}