import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import './App.css';
import LegacyDashboard from './LegacyDashboard.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <LegacyDashboard />
  </StrictMode>,
);
