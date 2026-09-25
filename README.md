# RetinaGuard

RetinaGuard is a React/Vite screening workspace backed by an Express API, PostgreSQL, Cloudinary image storage, and an ONNX inference worker.

The dashboard, review queue, report list, and case details are connected to the API. The new-screening page is intentionally a placeholder while the separate patient-form components and Clerk authentication are being integrated. No frontend image upload is currently wired to that page.

## Local development

1. Install Node.js 20 or later and Python 3.10 or later.
2. Create a PostgreSQL database and a Cloudinary account.
3. Copy `backend/.env.example` to `backend/.env` and set the database and Cloudinary values. Keep real credentials local; do not commit `.env` files.
4. Copy `frontend/.env.example` to `frontend/.env`.
5. Install and start the API:

   ```powershell
   cd backend
   npm ci
   python -m pip install -r requirements.txt
   npm run dev
   ```

6. In another terminal, start the frontend:

   ```powershell
   cd frontend
   npm ci
   npm run dev
   ```

The API listens on port `8000` by default, and the Vite app is available at `http://localhost:5173`. The API creates its tables at startup and exposes `/health` for health checks.

The inference worker needs a real `dr_model.onnx` file. Place it in `backend/` or set `DR_MODEL_PATH` to its local path. If the model is missing, processing fails explicitly rather than returning a simulated clinical result.

## Deployment outline

Deploy the API and frontend as separate services:

- **API service:** use `backend` as the service root, `bash render-build.sh` as the build command, and `npm start` as the start command. Configure `DATABASE_URL`, Cloudinary credentials, the model file/path, and `FRONTEND_URL` in the hosting provider's secret/environment settings. Set `DOCTOR_DATABASE_URL` only if reports should use a different PostgreSQL database.
- **Frontend static site:** use `frontend` as the service root, `npm ci && npm run build` as the build command, and `dist` as the publish directory. Set `VITE_API_BASE_URL` to the deployed API origin followed by `/api`, for example `https://api.example.com/api`.
- Set the API's `FRONTEND_URL` to the exact deployed frontend origin so browser requests pass CORS checks. Use HTTPS and hosted PostgreSQL SSL settings in production.

The API currently has no Clerk authentication; do not expose patient records or screening endpoints publicly. Add and verify server-side Clerk token validation and authorization before deploying with real patient data. The static frontend also needs a frontend host that serves `index.html` for direct navigation if client-side paths are introduced later.
