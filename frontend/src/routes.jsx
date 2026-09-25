import { Routes, Route } from "react-router-dom";

import Home from "./pages/public/Home";
import Login from "./pages/auth/Login";
import Register from "./pages/auth/Register";
import DoctorRegister from "./pages/auth/DoctorRegister";
import RegistrationPending from "./pages/auth/RegistrationPending";

import OperatorDashboard from "./pages/operator/OperatorDashboard";
import DoctorDashboard from "./pages/doctor/DoctorDashboard";

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/login/ophthalmologist" element={<Login initialRole="ophthalmologist" />} />
      <Route path="/login/doctor" element={<Login initialRole="ophthalmologist" />} />
      <Route path="/login/operator" element={<Login initialRole="operator" />} />
      <Route path="/register" element={<Register />} />
      <Route path="/register/doctor" element={<DoctorRegister />} />
      <Route path="/registration-pending" element={<RegistrationPending />} />
      <Route path="/operator" element={<OperatorDashboard />} />
      <Route path="/doctor" element={<DoctorDashboard />} />
    </Routes>
  );
}