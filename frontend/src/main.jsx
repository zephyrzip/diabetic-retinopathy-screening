import React from 'react';
import { createRoot } from 'react-dom/client';
import { Auth0Provider } from '@auth0/auth0-react';
import App from './App.jsx';
import './styles.css';

const domain = import.meta.env.VITE_AUTH0_DOMAIN;
const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID;

function ConfigurationMessage() {
  return <main className="config-message"><h1>Auth0 setup needed</h1><p>Copy <code>frontend/.env.example</code> to <code>frontend/.env</code> and add your Auth0 domain and client ID.</p></main>;
}

createRoot(document.getElementById('root')).render(
  domain && clientId ? (
    <Auth0Provider
      domain={domain}
      clientId={clientId}
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: import.meta.env.VITE_AUTH0_AUDIENCE,
        scope: 'openid profile email',
      }}
      cacheLocation="memory"
    >
      <App />
    </Auth0Provider>
  ) : <ConfigurationMessage />,
);
