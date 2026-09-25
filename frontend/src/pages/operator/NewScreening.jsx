import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { startScreening, uploadScreening } from '../../services/screenings';
import '../../styles/operator.css';

const initialValues = { patient_id: '', patient_name: '', age: '', gender: 'unknown', technician_id: '' };

export default function NewScreening() {
  const navigate = useNavigate();
  const [values, setValues] = useState(initialValues);
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  function updateValue(event) {
    setValues((current) => ({ ...current, [event.target.name]: event.target.value }));
  }

  async function submit(event) {
    event.preventDefault();
    if (!file) {
      setError('Choose a retinal image before creating the screening.');
      return;
    }
    setSubmitting(true);
    setError('');
    try {
      const created = await uploadScreening({ ...values, file });
      await startScreening(created.screening_id);
      navigate(`/operator/result/${created.screening_id}`);
    } catch (requestError) {
      setError(requestError.message);
      setSubmitting(false);
    }
  }

  return (
    <div className="operator-layout">
      <aside className="operator-sidebar">
        <div className="operator-logo"><span>◉</span><strong>DR-Screen AI</strong></div>
        <nav className="operator-nav">
          <Link to="/operator" className="operator-nav-item"><span>⌂</span>Dashboard</Link>
          <Link to="/operator/new-screening" className="operator-nav-item active"><span>＋</span>New Screening</Link>
        </nav>
      </aside>
      <main className="operator-main">
        <header className="operator-topbar"><div><p className="operator-page-label">SCREENING INTAKE</p><h1>New retinal screening</h1></div></header>
        <section className="operator-panel" style={{ maxWidth: 760 }}>
          <div className="panel-header"><div><span className="panel-label">PATIENT AND IMAGE</span><h2>Create and process a screening</h2></div></div>
          <form onSubmit={submit} style={{ display: 'grid', gap: 16, paddingTop: 12 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <label>Patient ID<input name="patient_id" value={values.patient_id} onChange={updateValue} required /></label>
              <label>Patient name<input name="patient_name" value={values.patient_name} onChange={updateValue} required /></label>
              <label>Age<input name="age" type="number" min="0" max="120" value={values.age} onChange={updateValue} required /></label>
              <label>Gender<select name="gender" value={values.gender} onChange={updateValue}><option value="unknown">Prefer not to say</option><option value="female">Female</option><option value="male">Male</option><option value="other">Other</option></select></label>
              <label>Technician ID<input name="technician_id" value={values.technician_id} onChange={updateValue} required /></label>
              <label>Fundus image<input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => setFile(event.target.files?.[0] || null)} required /></label>
            </div>
            <p style={{ margin: 0, color: '#64748b', fontSize: 13 }}>JPEG, PNG, or WEBP only; maximum 10 MB. Submitting uploads the image and starts AI processing.</p>
            {error && <p role="alert" style={{ margin: 0, color: '#b91c1c' }}>{error}</p>}
            <div><button className="operator-primary-button" type="submit" disabled={submitting}>{submitting ? 'Creating screening…' : 'Upload and start screening →'}</button></div>
          </form>
        </section>
      </main>
    </div>
  );
}
