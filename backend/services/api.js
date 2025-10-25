const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const fetchFixtures = async () => {
  const response = await fetch(`${API_BASE_URL}/fixtures`);
  return response.json();
};

export const fetchTeams = async () => {
  const response = await fetch(`${API_BASE_URL}/teams`);
  return response.json();
};

export const predictMatch = async (homeTeam, awayTeam) => {
  const response = await fetch(`${API_BASE_URL}/predictions/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ homeTeam, awayTeam })
  });
  return response.json();
};