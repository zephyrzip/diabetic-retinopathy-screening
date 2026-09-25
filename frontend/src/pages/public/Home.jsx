import { Link } from "react-router-dom";
import heroBackground from "../../assets/images/hero-background.png";
import { useEffect, useRef, useState } from "react";
import LoginRoleModal from "../../components/auth/LoginRoleModal";

export default function Home() {
  const [loginModalOpen, setLoginModalOpen] = useState(false);
  const [navDropdownOpen, setNavDropdownOpen] = useState(false);

  const featuresRef = useRef(null);
  const workflowRef = useRef(null);
  const ctaRef = useRef(null);

  // ==============================
  // FEATURES ANIMATION
  // ==============================
  useEffect(() => {
    const section = featuresRef.current;

    if (!section) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          section.classList.add("features-visible");
          observer.disconnect();
        }
      },
      {
        threshold: 0.2,
      }
    );

    observer.observe(section);

    return () => observer.disconnect();
  }, []);

  // ==============================
  // WORKFLOW ANIMATION
  // ==============================
  useEffect(() => {
    const section = workflowRef.current;

    if (!section) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          section.classList.add("workflow-visible");
          observer.disconnect();
        }
      },
      {
        threshold: 0.2,
      }
    );

    observer.observe(section);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
  const section = ctaRef.current;

  if (!section) return;

  const observer = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting) {
        section.classList.add("cta-visible");
        observer.disconnect();
      }
    },
    {
      threshold: 0.25,
    }
  );

  observer.observe(section);

  return () => observer.disconnect();
}, []);

  return (
    <div className="home-page">

      {/* =========================
          NAVBAR
      ========================= */}
      <header className="navbar">

        <div className="logo">
          <span className="logo-icon">◉</span>
          <span>DR-Screen AI</span>
        </div>

        <nav className="nav-links">
          <a href="#home">Home</a>
          <a href="#about">About</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#contact">Contact</a>
        </nav>

        <div
          className="navbar-login-container"
          onMouseEnter={() => setNavDropdownOpen(true)}
          onMouseLeave={() => setNavDropdownOpen(false)}
        >
          <button
            type="button"
            className="login-button navbar-login-trigger"
            onClick={() => setNavDropdownOpen((prev) => !prev)}
            aria-haspopup="true"
            aria-expanded={navDropdownOpen}
          >
            <span>Professional Login</span>
            <span className="dropdown-arrow-icon">▾</span>
          </button>

          {/* Hover Dropdown */}
          <div className={`navbar-login-dropdown ${navDropdownOpen ? "dropdown-visible" : ""}`}>
            <div className="dropdown-header">
              <span className="dropdown-label">Select Login Profile</span>
            </div>

            {/* Option 1: Ophthalmologist Profile */}
            <Link
              to="/login/ophthalmologist"
              className="dropdown-profile-item"
              onClick={() => setNavDropdownOpen(false)}
            >
              <div className="dropdown-item-icon doctor-icon">
                <span>🩺</span>
              </div>
              <div className="dropdown-item-content">
                <div className="dropdown-item-title-row">
                  <strong>Ophthalmologist Profile</strong>
                  <span className="dropdown-role-tag doctor-tag">Specialist</span>
                </div>
                <p>Review AI classifications, Grad-CAM maps & approve clinical diagnoses</p>
              </div>
              <span className="dropdown-item-arrow">→</span>
            </Link>

            {/* Option 2: Screening Operator Profile */}
            <Link
              to="/login/operator"
              className="dropdown-profile-item"
              onClick={() => setNavDropdownOpen(false)}
            >
              <div className="dropdown-item-icon operator-icon">
                <span>🔬</span>
              </div>
              <div className="dropdown-item-content">
                <div className="dropdown-item-title-row">
                  <strong>Screening Operator Profile</strong>
                  <span className="dropdown-role-tag operator-tag">Screening</span>
                </div>
                <p>Capture fundus images, check quality & generate AI predictions</p>
              </div>
              <span className="dropdown-item-arrow">→</span>
            </Link>

            <div className="dropdown-footer">
              <span>New account?</span>
              <div className="dropdown-register-links">
                <Link to="/register/doctor" onClick={() => setNavDropdownOpen(false)}>Doctor Signup</Link>
                <span className="sep">•</span>
                <Link to="/register" onClick={() => setNavDropdownOpen(false)}>Operator Signup</Link>
              </div>
            </div>
          </div>
        </div>

      </header>


      {/* =========================
          MAIN
      ========================= */}
      <main>

        {/* =========================
            HERO
        ========================= */}
        <section
          className="hero"
          id="home"
          style={{
            backgroundImage: `url(${heroBackground})`,
          }}
        >

          <div className="hero-content">

            <div className="hero-badge">
              AI-Powered Retinal Screening
            </div>

            <h1>
              Explainable AI for
              <span> Diabetic Retinopathy </span>
              Screening
            </h1>

            <p>
              An intelligent screening platform that analyzes retinal
              images, detects diabetic retinopathy, and provides
              explainable AI insights to support healthcare professionals.
            </p>

            <div className="hero-buttons">

              <button
                type="button"
                className="primary-button"
                onClick={() => setLoginModalOpen(true)}
              >
                Professional Login →
              </button>

              <a
                href="#how-it-works"
                className="secondary-button"
              >
                How It Works
              </a>

            </div>

          </div>

        </section>


        {/* =========================
            FEATURES
        ========================= */}
        <section
          className="features"
          id="about"
          ref={featuresRef}
        >

          <div className="section-heading features-heading">

            <span>CORE FEATURES</span>

            <h2>
              Intelligent Retinal Screening
            </h2>

            <p>
              Designed to support early detection and remote
              ophthalmologist review.
            </p>

          </div>


          <div className="feature-grid">

            {/* CARD 1 */}
            <div className="feature-card feature-card-1">

              <div className="feature-icon">
                ◎
              </div>

              <h3>
                Image Quality Assessment
              </h3>

              <p>
                Automatically checks retinal image quality and
                identifies images that require recapture.
              </p>

            </div>


            {/* CARD 2 */}
            <div className="feature-card feature-card-2">

              <div className="feature-icon">
                ◈
              </div>

              <h3>
                DR Severity Classification
              </h3>

              <p>
                Classifies diabetic retinopathy across five severity
                levels from No DR to Proliferative DR.
              </p>

            </div>


            {/* CARD 3 */}
            <div className="feature-card feature-card-3">

              <div className="feature-icon">
                ✦
              </div>

              <h3>
                Explainable AI
              </h3>

              <p>
                Uses Grad-CAM and lesion evidence to show regions
                contributing to the AI prediction.
              </p>

            </div>


            {/* CARD 4 */}
            <div className="feature-card feature-card-4">

              <div className="feature-icon">
                ◉
              </div>

              <h3>
                Ophthalmologist Review
              </h3>

              <p>
                Enables remote specialists to review referred cases
                and provide the final clinical assessment.
              </p>

            </div>

          </div>

        </section>


        {/* =========================
            WORKFLOW
        ========================= */}
        <section
          className="how-it-works"
          id="how-it-works"
          ref={workflowRef}
        >

          <div className="section-heading workflow-heading">

            <span>
              WORKFLOW
            </span>

            <h2>
              How the System Works
            </h2>

          </div>


          <div className="workflow">

            {/* STEP 1 */}
            <div className="workflow-step workflow-step-1">

              <div className="step-number">
                01
              </div>

              <h3>
                Capture
              </h3>

              <p>
                Capture or upload a retinal fundus image.
              </p>

            </div>


            <div className="workflow-line workflow-line-1"></div>


            {/* STEP 2 */}
            <div className="workflow-step workflow-step-2">

              <div className="step-number">
                02
              </div>

              <h3>
                Check Quality
              </h3>

              <p>
                The system evaluates focus, illumination and
                field of view.
              </p>

            </div>


            <div className="workflow-line workflow-line-2"></div>


            {/* STEP 3 */}
            <div className="workflow-step workflow-step-3">

              <div className="step-number">
                03
              </div>

              <h3>
                AI Screening
              </h3>

              <p>
                The AI predicts the DR severity and referable status.
              </p>

            </div>


            <div className="workflow-line workflow-line-3"></div>


            {/* STEP 4 */}
            <div className="workflow-step workflow-step-4">

              <div className="step-number">
                04
              </div>

              <h3>
                Explain & Review
              </h3>

              <p>
                AI evidence is displayed for healthcare professional
                review.
              </p>

            </div>

          </div>

        </section>


        {/* =========================
            CTA
        ========================= */}
        <section
  className="cta"
  id="cta-section"
  ref={ctaRef}
>

  <div className="cta-content">
    <h2>Ready to begin screening?</h2>

    <p>
      Access the healthcare professional dashboard to start a new
      retinal screening.
    </p>
  </div>

  <button
    type="button"
    className="primary-button cta-button"
    onClick={() => setLoginModalOpen(true)}
  >
    Professional Login →
  </button>

</section>

      </main>


     {/* =========================
    FOOTER
========================= */}
<footer className="footer">

  <div className="footer-main">

    <div className="footer-brand">
      <div className="footer-logo">
        <span className="footer-logo-icon">◉</span>
        <span>DR-Screen AI</span>
      </div>

      <p>
        Explainable AI for Diabetic Retinopathy Screening
      </p>
    </div>

    <div className="footer-purpose">
      <span className="footer-label">SCREENING PLATFORM</span>

      <p>
        AI-assisted retinal screening with
        human-in-the-loop clinical review.
      </p>

      <button
        type="button"
        onClick={() => setLoginModalOpen(true)}
        style={{
          marginTop: "12px",
          background: "rgba(32, 166, 179, 0.15)",
          border: "1px solid rgba(32, 166, 179, 0.4)",
          color: "#20a6b3",
          padding: "7px 14px",
          borderRadius: "6px",
          fontSize: "12px",
          fontWeight: "600",
          cursor: "pointer",
          transition: "0.2s ease",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(32, 166, 179, 0.28)")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(32, 166, 179, 0.15)")}
      >
        Professional Login Portal →
      </button>
    </div>

  </div>


  <div className="footer-divider"></div>


  <div className="footer-bottom">

    <span>
      © 2026 DR-Screen AI. All rights reserved.
    </span>

    <span className="footer-status">
      <span className="footer-status-dot"></span>
      AI-assisted screening
    </span>

  </div>

</footer>

      {/* Login Role Selector Modal */}
      <LoginRoleModal
        isOpen={loginModalOpen}
        onClose={() => setLoginModalOpen(false)}
      />
    </div>
  );
}