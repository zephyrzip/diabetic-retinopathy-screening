# diabetic-retinopathy-screening
Explainable AI for Diabetic Retinopathy Screening in Rural India

Description->India has over 77 million diabetic adults - the second highest globally. Diabetic Retinopathy (DR) affects ~18% of this population and is a leading cause of preventable blindness. Early screening can prevent90% of vision loss, but India has only ~1 ophthalmologist per 100,000 rural population, making mass manual screening infeasible. Existing AI solutions function as black boxes, lack clinical validation rigor, and fail with variable image quality from portable fundus cameras in field conditions. A robust, explainable, and validated screening system is essential for deployment in primary healthcare centres across rural India.

Description:

Design a MATLAB-based retinal image analysis pipeline for automated DR screening addressing real-world deployment challenges:

1. Image Quality Assessment and Enhancement: Automatically evaluate fundus images for adequacy (focus, illumination, field of view). Apply adaptive enhancement (CLAHE, illumination normalization, denoising) for borderline images; reject ungradeable ones with recapture feedback.

2. Retinal Structure Segmentation: Extract clinically relevant structures - optic disc/fovea localization, vessel segmentation, microaneurysm detection, exudate segmentation, hemorrhage classification, and neovascularization detection.

3. DR Severity Grading: Classify using the International Clinical DR severity scale (Levels 0-4, from no DR to proliferative DR) with clinically acceptable sensitivity (>90%) and specificity (>85%) for referable DR (Level 2+).

4. Explainability Module: Implement Grad-CAM attention maps, lesion-level evidence correlated with clinical criteria, calibrated confidence scores, and automated annotated reports - enabling ophthalmologist validation in under 30 seconds for a human-in-theloop workflow.

5. Simulink Workflow Simulation: Model the telemedicine screening pipeline in Simulink - image acquisition rates, bandwidth constraints, processing throughput, and review capacity - to optimize resource allocation for district-level programs serving 100,000+ patients annually.

This problem demands clinical validation rigor, sub-pixel microaneurysm detection, and clinically meaningful explainability

• Tools: Image Processing Toolbox, Computer Vision Toolbox, Deep Learning Toolbox, Medical Imaging Toolbox, Simulink, Statistics and Machine Learning Toolbox Expected Solution: A working prototype demonstrating: DR classification with >90% sensitivity and >85% specificity for referable DR;

explainable Grad-CAM outputs rated as clinically useful; a Simulink model optimizing screening resource allocation; and validation against published benchmarks showing the integrated pipeline outperforms any single technique approach.

## MERN application: Auth0 sign-in

The repository includes an initial React/Vite frontend and Express API for secure sign-in. Auth0 handles user authentication, and the API validates Auth0 access tokens before returning protected resources. Patient records and screening workflows are not implemented yet.

### Configure Auth0

1. Create an Auth0 **Single Page Application** and an Auth0 **API**.
2. Set the API identifier to `https://dr-screening-api` (or choose another identifier and use it consistently in both `.env` files). Keep the API signing algorithm as RS256.
3. In the SPA settings, add `http://localhost:5173` to Allowed Callback URLs, Allowed Logout URLs, and Allowed Web Origins.
4. Copy `frontend/.env.example` to `frontend/.env` and set the Auth0 domain and SPA client ID. The domain is the tenant domain without `https://` or a trailing slash. Set the audience to the API identifier.
5. Copy `backend/.env.example` to `backend/.env`. Set `AUTH0_ISSUER_BASE_URL` to the tenant URL with `https://` and a trailing slash, and set `AUTH0_AUDIENCE` to the same API identifier. Keep the frontend origin at `http://localhost:5173` for local development.

### Run locally

Use two terminals from the repository root:

```sh
cd backend
npm install
npm run dev
```

```sh
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The login page redirects to Auth0. Once signed in, the client obtains an access token for the API and calls the protected `/api/me` endpoint. `/api/health` is public for availability checks.

Do not commit `.env` files or put an Auth0 client secret in the browser app. The frontend uses Auth0's in-memory token cache; a fresh page load may require signing in again. Production deployments need their own callback/origin allowlists, HTTPS URLs, and secret management.

### API authorization roles

The API authorizes using permission strings from the verified Auth0 access token. It does not trust frontend-only role checks. Configure the same permission strings under **Applications > APIs > your API > Permissions**, enable **RBAC** and **Add Permissions in the Access Token**, then create these Auth0 roles and assign their permissions:

| Auth0 role | API permissions |
| --- | --- |
| Field technician | `images:capture`, `images:quality:grade`, `images:submit` |
| Ophthalmologist | `cases:read`, `cases:referable:confirm`, `evidence:gradcam:read` |
| Program admin | `program:caseload:read`, `referrals:read`, `referrals:manage`, `screening-data:read`, `screening-data:manage` |

Assign each user the appropriate Auth0 role. `/api/me` returns only recognized permissions from that user's verified token so the frontend can display the right workspace; the backend remains responsible for enforcing them on every feature route. Use `requirePermissions('cases:referable:confirm')` after `requireAuth` on an ophthalmologist-only route. A missing permission claim is denied by default. Do not rely on frontend route hiding as access control.

The permission `images:submit` grants the action only. When image submission is implemented, the API must also check the image's persisted quality result is `gradeable` before accepting it; a role permission cannot enforce that record-level condition. Program admin permissions cover program operations and screening data, not clinical referable-case confirmation. Add narrower permissions if their data scope needs to expand.

For the frontend team: keep requesting an access token for this API's audience and use the token's `permissions` returned by `/api/me` for navigation. Never treat UI checks as authorization; send the access token with requests and let the API apply permission middleware. After changing Auth0 roles or API RBAC settings, obtain a fresh access token.
