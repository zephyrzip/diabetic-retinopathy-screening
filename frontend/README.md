# RetinaGuard frontend

The frontend contains two entry points that serve different needs:

- `index.html` opens the routed screening experience, including public, operator, and doctor pages.
- `dashboard.html` opens the API-connected dashboard for screening cases, review queue, and reports.

Start the Vite server with `npm run dev` from this directory, then open `/` or `/dashboard.html`. The API-connected dashboard uses `http://localhost:8000/api` in development by default. Set `VITE_API_BASE_URL` in `frontend/.env` to use another API URL.

Build the frontend with `npm run build`. Both entry points are included in the production build.
