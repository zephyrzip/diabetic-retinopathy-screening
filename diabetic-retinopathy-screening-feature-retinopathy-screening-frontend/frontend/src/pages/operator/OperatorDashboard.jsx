import { Link } from "react-router-dom";
import "../../styles/operator.css";

export default function OperatorDashboard() {
  const recentScreenings = [
    {
      id: "P-001",
      date: "25 Sep 2026",
      time: "10:30 AM",
      level: "No DR",
      status: "Non-Referable",
    },
    {
      id: "P-002",
      date: "25 Sep 2026",
      time: "11:10 AM",
      level: "Moderate DR",
      status: "Referable",
    },
    {
      id: "P-003",
      date: "25 Sep 2026",
      time: "11:40 AM",
      level: "Severe DR",
      status: "Referable",
    },
    {
      id: "P-004",
      date: "25 Sep 2026",
      time: "12:15 PM",
      level: "No DR",
      status: "Non-Referable",
    },
  ];

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
            to="/operator/history"
            className="operator-nav-item"
          >
            <span>▤</span>
            Screening History
          </Link>

          <Link
            to="/operator/reports"
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
              <strong>24</strong>
              <small>Today</small>
            </div>

          </div>


          <div className="operator-stat-card">

            <div className="stat-icon referable">
              !
            </div>

            <div>
              <span>Referable Cases</span>
              <strong>7</strong>
              <small>Needs review</small>
            </div>

          </div>


          <div className="operator-stat-card">

            <div className="stat-icon quality">
              ✓
            </div>

            <div>
              <span>Good Quality</span>
              <strong>21</strong>
              <small>87.5% of images</small>
            </div>

          </div>


          <div className="operator-stat-card">

            <div className="stat-icon recapture">
              ↻
            </div>

            <div>
              <span>Recapture Required</span>
              <strong>3</strong>
              <small>Image quality</small>
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

            <Link to="/operator/history">
              View All →
            </Link>

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

                {recentScreenings.map((screening) => (

                  <tr key={screening.id}>

                    <td>
                      <strong>
                        {screening.id}
                      </strong>
                    </td>

                    <td>
                      {screening.date}
                    </td>

                    <td>
                      {screening.time}
                    </td>

                    <td>
                      <span className="dr-level">
                        {screening.level}
                      </span>
                    </td>

                    <td>

                      <span
                        className={
                          screening.status === "Referable"
                            ? "status-badge referable-status"
                            : "status-badge safe-status"
                        }
                      >
                        {screening.status}
                      </span>

                    </td>

                    <td>

                      <button
                        className="view-button"
                        type="button"
                      >
                        View
                      </button>

                    </td>

                  </tr>

                ))}

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
              to="/operator/history"
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
              to="/operator/reports"
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