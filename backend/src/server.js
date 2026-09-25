import 'dotenv/config';
import cors from 'cors';
import express from 'express';
import { auth } from 'express-oauth2-jwt-bearer';
import helmet from 'helmet';
import { API_PERMISSIONS } from './authz.js';

const { AUTH0_ISSUER_BASE_URL, AUTH0_AUDIENCE } = process.env;
if (!AUTH0_ISSUER_BASE_URL || !AUTH0_AUDIENCE) {
  throw new Error('AUTH0_ISSUER_BASE_URL and AUTH0_AUDIENCE must be configured.');
}

const app = express();
app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_ORIGIN ?? 'http://localhost:5173' }));
app.use(express.json());

const requireAuth = auth({
  issuerBaseURL: AUTH0_ISSUER_BASE_URL,
  audience: AUTH0_AUDIENCE,
  tokenSigningAlg: 'RS256',
});

app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));

// All application data routes should use this middleware. The API verifies the
// access token itself; a signed-in browser session alone is not sufficient.
app.get('/api/me', requireAuth, (req, res) => {
  const permissions = req.auth.payload.permissions;
  res.json({
    sub: req.auth.payload.sub,
    permissions: Array.isArray(permissions)
      ? permissions.filter((permission) => API_PERMISSIONS.includes(permission))
      : [],
  });
});

app.use((err, _req, res, _next) => {
  if (err.name === 'UnauthorizedError' || err.status === 401) {
    return res.status(401).json({ error: 'A valid access token is required.' });
  }
  console.error(err);
  return res.status(500).json({ error: 'Internal server error.' });
});

const port = Number(process.env.PORT ?? 4000);
app.listen(port, () => console.log(`DR screening API listening on http://localhost:${port}`));
