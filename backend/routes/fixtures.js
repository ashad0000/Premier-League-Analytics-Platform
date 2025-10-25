// backend/routes/fixtures.js
const express = require('express');
const router = express.Router();

// ✅ Sample 2025/26 Fixtures
const fixtures = [
  { id: 1, homeTeam: 'Arsenal', awayTeam: 'Liverpool', date: '2025-11-02', stadium: 'Emirates Stadium' },
  { id: 2, homeTeam: 'Manchester City', awayTeam: 'Manchester United', date: '2025-11-09', stadium: 'Etihad Stadium' },
  { id: 3, homeTeam: 'Tottenham Hotspur', awayTeam: 'Chelsea', date: '2025-11-16', stadium: 'Tottenham Hotspur Stadium' },
  { id: 4, homeTeam: 'Aston Villa', awayTeam: 'Newcastle United', date: '2025-11-23', stadium: 'Villa Park' },
  { id: 5, homeTeam: 'Liverpool', awayTeam: 'Arsenal', date: '2025-11-30', stadium: 'Anfield' }
];

// GET all fixtures
router.get('/', (req, res) => {
  res.json(fixtures);
});

// GET fixture by ID
router.get('/:id', (req, res) => {
  const fixture = fixtures.find(f => f.id === parseInt(req.params.id));
  fixture ? res.json(fixture) : res.status(404).json({ message: 'Fixture not found' });
});

module.exports = router;
