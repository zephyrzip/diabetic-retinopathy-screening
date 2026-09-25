import { Routes, Route } from "react-router-dom";

import Home from "./pages/public/Home";
import Login from "./pages/auth/Login";
import Register from "./pages/auth/Register";
import DoctorRegister from "./pages/auth/DoctorRegister";
import RegistrationPending from "./pages/auth/RegistrationPending";

import OperatorDashboard from "./pages/operator/OperatorDashboard";
import NewScreening from "./pages/operator/NewScreening";
import Screening from "./pages/operator/Screening";
import ScreeningResult from "./pages/operator/ScreeningResult";
import Explainability from "./pages/operator/Explainability";

import DoctorDashboard from "./pages/doctor/DoctorDashboard";
import ReviewQueuePage from "./pages/doctor/ReviewQueuePage";
import PatientReviewPage from "./pages/doctor/PatientReviewPage";

function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<Home />} />

      {/* Authentication - General & Role-Specific Logins */}
      <Route path="/login" element={<Login />} />
      <Route
        path="/login/ophthalmologist"
        element={<Login initialRole="ophthalmologist" />}
      />
      <Route
        path="/login/doctor"
        element={<Login initialRole="ophthalmologist" />}
      />
      <Route
        path="/login/operator"
        element={<Login initialRole="operator" />}
      />

      {/* Authentication - Registration Portals */}
      <Route path="/register" element={<Register />} />
      <Route path="/register/operator" element={<Register />} />
      <Route path="/register/doctor" element={<DoctorRegister />} />
      <Route path="/register/ophthalmologist" element={<DoctorRegister />} />
      <Route path="/registration-pending" element={<RegistrationPending />} />

      {/* Screening Operator */}
      <Route path="/operator" element={<OperatorDashboard />} />
      <Route path="/operator/new-screening" element={<NewScreening />} />
      <Route path="/operator/screening" element={<Screening />} />
      <Route path="/operator/result/:screeningId" element={<ScreeningResult />} />
      <Route path="/operator/explainability" element={<Explainability />} />

      {/* Ophthalmologist / Clinical */}
      <Route path="/doctor" element={<DoctorDashboard />} />
      <Route path="/doctor/reviews" element={<ReviewQueuePage />} />
      <Route path="/doctor/review/:patientId" element={<PatientReviewPage />} />
    </Routes>
  );
}

export default App;
