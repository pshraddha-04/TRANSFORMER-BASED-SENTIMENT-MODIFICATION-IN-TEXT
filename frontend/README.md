# Frontend

This folder contains the React + Vite dashboard for the sentiment transformation API.

## Run locally

1. Start the backend:

```powershell
cd "C:\Users\vp326\Desktop\Sem6_2.0"
$env:LIGHTWEIGHT_MODE="1"
python main.py
```

2. Start the frontend:

```powershell
cd "C:\Users\vp326\Desktop\Sem6_2.0\frontend"
npm install
npm run dev
```

3. Open the UI in your browser:

```text
http://localhost:5173
```

## Notes
- The UI defaults to `VITE_API_BASE_URL=/api` and the Vite proxy rewrites `/api/*` to the backend root routes like `/health` and `/predict`.
- You can change the base URL in the UI at runtime.
- The Connection panel includes an optional API key field; when provided, requests include `X-API-Key`.
- The backend has CORS enabled for direct browser access when you want to point the UI at a deployed API URL.

## CI

- Root workflow `.github/workflows/ci.yml` validates this folder with `npm test` and `npm run build`.

## Troubleshooting
- If you still see `Not Found` while using `/api`, restart the frontend dev server so Vite reloads proxy config.
- As a fallback, set API base URL in the UI to `http://127.0.0.1:8000` and click **Save endpoint**.

