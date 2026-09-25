const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  || (import.meta.env.DEV ? 'http://localhost:8000/api' : '');

async function request(path, options = {}) {
  if (!apiBaseUrl) {
    throw new Error('Set VITE_API_BASE_URL to the deployed API URL.');
  }
  const response = await fetch(`${apiBaseUrl}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `The screening service returned ${response.status}.`);
  }
  return body;
}

export function listScreenings(signal) {
  return request('/screenings', { signal });
}

export function getScreening(id, signal) {
  return request(`/screenings/${id}`, { signal });
}

export function uploadScreening(values) {
  const formData = new FormData();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') formData.append(key, value);
  });
  return request('/screenings/upload', { method: 'POST', body: formData });
}

export function startScreening(id) {
  return request(`/screenings/${id}/process`, { method: 'POST' });
}

export function signScreeningReview(id, review) {
  return request(`/screenings/${id}/review`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(review),
  });
}

export function gradeLabel(grade) {
  return ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'][Number(grade)] || 'Awaiting AI result';
}

export function formatDate(value) {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)) : '—';
}
