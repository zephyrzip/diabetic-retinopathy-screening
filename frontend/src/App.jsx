import { useEffect, useState } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const apiUrl = import.meta.env.VITE_API_URL ?? 'http://localhost:4000';

function LoginPage({ onLogin, isLoading }) {
  return (
    <main className="page-shell">
      <header className="topbar"><a className="brand" href="#"><span className="brand-mark">R</span> RetinaCare</a><span className="topbar-note">Rural eye health, supported by AI</span></header>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow"><span className="status-dot" /> DIABETIC RETINOPATHY SCREENING</p>
          <h1>Earlier insight.<br /><em>Clearer futures.</em></h1>
          <p className="intro">A screening workspace for healthcare teams bringing retinal assessment closer to the communities that need it.</p>
          <button className="primary-button" onClick={onLogin} disabled={isLoading}>{isLoading ? 'Loading…' : 'Sign in to your workspace'}<span aria-hidden="true">→</span></button>
          <p className="secure-note"><span aria-hidden="true">⌑</span> Secure sign-in powered by Auth0</p>
        </div>
        <div className="visual" aria-label="Illustration of a retina scan">
          <div className="orbit orbit-one" /><div className="orbit orbit-two" />
          <div className="retina"><div className="retina-core" /><i className="vessel v1"/><i className="vessel v2"/><i className="vessel v3"/><i className="vessel v4"/><i className="vessel v5"/><i className="vessel v6"/></div>
          <div className="scan-label"><span className="scan-pulse"/> SCREENING READY</div>
          <span className="visual-caption">A clearer view of retinal health</span>
        </div>
      </section>
      <footer className="footer"><span>RETINACARE · SCREENING SUPPORT</span><span>For trained healthcare professionals</span></footer>
    </main>
  );
}

function Dashboard({ user, getAccessTokenSilently, onLogout }) {
  const [apiState, setApiState] = useState('loading');
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const token = await getAccessTokenSilently();
        const response = await fetch(`${apiUrl}/api/me`, { headers: { Authorization: `Bearer ${token}` } });
        if (!response.ok) throw new Error('API access was not authorized. Check Auth0 API settings and configuration.');
        const data = await response.json();
        if (active) setApiState(`Connected securely · ${data.sub}`);
      } catch (error) { if (active) setApiState(error.message); }
    })();
    return () => { active = false; };
  }, [getAccessTokenSilently]);

  return <main className="page-shell dashboard-shell"><header className="topbar"><a className="brand" href="#"><span className="brand-mark">R</span> RetinaCare</a><button className="text-button" onClick={onLogout}>Sign out</button></header><section className="welcome"><p className="eyebrow">SECURE WORKSPACE</p><h1>Welcome{user?.name ? `, ${user.name.split(' ')[0]}` : ''}.</h1><p className="intro">You’re signed in. Your screening workspace is ready for the next step.</p><div className="connection-card"><span className="status-dot"/><div><strong>API authorization</strong><p>{apiState}</p></div></div><div className="coming-card"><span className="coming-icon">＋</span><div><strong>Screening workspace</strong><p>Patient and image review tools will appear here as they are added to the project.</p></div></div></section><footer className="footer"><span>RETINACARE · SCREENING SUPPORT</span><span>For trained healthcare professionals</span></footer></main>;
}

export default function App() {
  const { isAuthenticated, isLoading, loginWithRedirect, logout, user, getAccessTokenSilently } = useAuth0();
  if (isLoading) return <main className="loading-screen"><span className="spinner"/><span>Preparing your secure workspace…</span></main>;
  if (!isAuthenticated) return <LoginPage onLogin={() => loginWithRedirect()} isLoading={isLoading}/>;
  return <Dashboard user={user} getAccessTokenSilently={getAccessTokenSilently} onLogout={() => logout({ logoutParams: { returnTo: window.location.origin } })}/>;
}
