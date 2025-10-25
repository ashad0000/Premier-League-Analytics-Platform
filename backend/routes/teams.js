const express = require('express');
const router = express.Router();

const teams = [
  { id: 1, name: 'Arsenal', stadium: 'Emirates Stadium', city: 'London' },
  { id: 2, name: 'Aston Villa', stadium: 'Villa Park', city: 'Birmingham' },
];

router.get('/', (req, res) => res.json(teams));

module.exports = router;
