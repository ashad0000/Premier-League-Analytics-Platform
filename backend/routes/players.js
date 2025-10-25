// backend/routes/players.js
const express = require('express');
const router = express.Router();

// ✅ 2025/26 Premier League Players (Updated)
const players = [
  { id: 1, name: 'Erling Haaland', team: 'Manchester City', position: 'Forward' },
  { id: 2, name: 'Bukayo Saka', team: 'Arsenal', position: 'Winger' },
  { id: 3, name: 'Mohamed Salah', team: 'Liverpool', position: 'Forward' },
  { id: 4, name: 'Cole Palmer', team: 'Chelsea', position: 'Midfielder' },
  { id: 5, name: 'Bruno Fernandes', team: 'Manchester United', position: 'Midfielder' },
  { id: 6, name: 'Richarlison', team: 'Tottenham Hotspur', position: 'Forward' },
  { id: 7, name: 'Ollie Watkins', team: 'Aston Villa', position: 'Striker' },
  { id: 8, name: 'Alexander Isak', team: 'Liverpool', position: 'Forward' },
  { id: 9, name: 'James Maddison', team: 'Tottenham Hotspur', position: 'Midfielder' },
  { id: 10, name: 'Virgil van Dijk', team: 'Liverpool', position: 'Defender' }
];

// GET all players
router.get('/', (req, res) => {
  res.json(players);
});

// GET player by ID
router.get('/:id', (req, res) => {
  const player = players.find(p => p.id === parseInt(req.params.id));
  player ? res.json(player) : res.status(404).json({ message: 'Player not found' });
});

module.exports = router;
