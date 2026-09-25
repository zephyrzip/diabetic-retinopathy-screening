import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import "../../styles/operator.css";
import { formatDate, gradeLabel, listScreenings } from "../../services/screenings";

export default function OperatorDashboard() {
  const [screenings, setScreenings] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    listScreenings(controller.signal).then((data) => setScreenings(data.screenings || [])).catch((requestError) => {
      if (requestError.name !== "AbortError") setError(requestError.message);
    });
    return () => controller.abort();
  }, []);

  const totals = useMemo(() => ({
    total: screenings.length,
    referable: screenings.filter((screening) => screening.is_referable).length,
    complete: screenings.filter((screening) => screening.status === "COMPLETED").length,
    failed: screenings.filter((screening) => screening.status === "FAILED").length,
  }), [screenings]);

  return (
    <div className="operator-layout">

      {/* =========================
          SIDEBAR
      ========================= */}

      <aside className="operator-sidebar">

        <div className="operator-logo">
          <span>◉</span>
          <strong>DR-Screen AI</strong>
        </div>

        <nav className="operator-nav">

          <Link
            to="/operator"
            className="operator-nav-item active"
          >
            <span>⌂</span>
            Dashboard
          </Link>

          <Link
            to="/operator/new-screening"
            className="operator-nav-item"
          >
            <span>＋</span>
            New Screening
          </Link>

          <Link
            to="/operator"
            className="operator-nav-item"
          >
            <span>▤</span>
            Screening History
          </Link>

          <Link
            to="/operator"
            className="operator-nav-item"
          >
            <span>▧</span>
            Reports
          </Link>

        </nav>

        <div className="operator-sidebar-bottom">

          <Link
            to="/operator/settings"
            className="operator-nav-item"
          >
            <span>⚙</span>
            Settings
          </Link>

          <Link
            to="/login"
            className="operator-nav-item logout"
          >
            <span>↪</span>
            Logout
          </Link>

        </div>

      </aside>


      {/* =========================
          MAIN AREA
      ========================= */}

      <main className="operator-main">

        {/* TOP BAR */}

        <header className="operator-topbar">

          <div>
            <p className="operator-page-label">
              SCREENING CENTER
            </p>

            <h1>
              Operator Dashboard
            </h1>
          </div>

          <div className="operator-user">

            <div className="notification">
              🔔
              <span></span>
            </div>

            <div className="operator-avatar">
              OP
            </div>

            <div className="operator-user-info">
              <strong>Screening Operator</strong>
              <small>Healthcare Professional</small>
            </div>

          </div>

        </header>


        {/* =========================
            WELCOME
        ========================= */}

        <section className="operator-welcome">

          <div>

            <span className="welcome-label">
              TODAY'S OVERVIEW
            </span>

            <h2>
              Welcome back, Operator
            </h2>

            <p>
              Manage retinal screenings and monitor today's
              screening activity.
            </p>

          </div>

          <Link
            to="/operator/new-screening"
            className="operator-primary-button"
          >
            + Start New Screening
          </Link>

        </section>


        {/* =========================
            STATISTICS
        ========================= */}

        <section className="operator-stats">

          <div className="operator-stat-card">

            <div className="stat-icon">
              ◉
            </div>

            <div>
              <span>Total Screenings</span>
              <strong>{totals.total}</strong>
              <small>All recorded screenings</small>
            </div>

          </div>


          <div className="operator-stat-card">

            <div className="stat-icon referable">
              !
            </div>

            <div>
              <span>Referable Cases</span>
              <strong>{totals.referable}</strong>
              <small>Needs review</small>
            </div>

          </div>


          <div className="operator-stat-card">

            <div className="stat-icon quality">
              ✓
            </div>

            <div>
              <span>Good Quality</span>
              <strong>{totals.complete}</strong>
              <small>AI processing completed</small>
            </div>

          </div>


          <div className="operator-stat-card">

            <div className="stat-icon recapture">
              ↻
            </div>

            <div>
              <span>Recapture Required</span>
              <strong>{totals.failed}</strong>
              <small>Needs follow-up</small>
            </div>

          </div>

        </section>


        {/* =========================
            RECENT SCREENINGS
        ========================= */}

        <section className="operator-panel">

          <div className="panel-header">

            <div>
              <span className="panel-label">
                SCREENING ACTIVITY
              </span>

              <h2>
                Recent Screenings
              </h2>
            </div>

            <Link to="/operator/new-screening">Start new →</Link>

          </div>


          <div className="screening-table-wrapper">

            <table className="screening-table">

              <thead>

                <tr>
                  <th>Patient ID</th>
                  <th>Date</th>
                  <th>Time</th>
                  <th>AI Result</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>

              </thead>

              <tbody>

                {screenings.slice(0, 10).map((screening) => (

                  <tr key={screening.screening_id}>

                    <td>
                      <strong>
                        {screening.patient_id}
                      </strong>
                    </td>

                    <td>
                      {formatDate(screening.created_at).split(',')[0]}
                    </td>

                    <td>
                      {formatDate(screening.created_at).split(',')[1]?.trim() || '—'}
                    </td>

                    <td>
                      <span className="dr-level">
                        {screening.status === 'COMPLETED' ? gradeLabel(screening.ai_grade) : screening.status}
                      </span>
                    </td>

                    <td>

                      <span
                        className={
                          screening.is_referable
                            ? "status-badge referable-status"
                            : "status-badge safe-status"
                        }
                      >
                        {screening.is_referable == null ? screening.status : screening.is_referable ? 'Referable' : 'Non-referable'}
                      </span>

                    </td>

                    <td>

                      <Link className="view-button" to={`/operator/result/${screening.screening_id}`}>View</Link>

                    </td>

                  </tr>

                ))}
                {!screenings.length && <tr><td colSpan="6">{error || 'No screenings have been created yet.'}</td></tr>}

              </tbody>

            </table>

          </div>

        </section>


        {/* =========================
            QUICK ACTIONS
        ========================= */}

        <section className="quick-actions">

          <h2>
            Quick Actions
          </h2>

          <div className="quick-action-grid">

            <Link
              to="/operator/new-screening"
              className="quick-action-card"
            >
              <span>＋</span>

              <div>
                <strong>
                  New Screening
                </strong>

                <p>
                  Start a new retinal screening
                </p>
              </div>

              <b>→</b>
            </Link>


            <Link
              to="/operator"
              className="quick-action-card"
            >
              <span>▤</span>

              <div>
                <strong>
                  Screening History
                </strong>

                <p>
                  View previous screening results
                </p>
              </div>

              <b>→</b>
            </Link>


            <Link
              to="/operator"
              className="quick-action-card"
            >
              <span>▧</span>

              <div>
                <strong>
                  Reports
                </strong>

                <p>
                  View and download reports
                </p>
              </div>

              <b>→</b>
            </Link>

          </div>

        </section>

      </main>

    </div>
  );
}
