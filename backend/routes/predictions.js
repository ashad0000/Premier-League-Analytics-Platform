const express = require('express');
const router = express.Router();
const { runPrediction } = require('../services/mlService');

router.post('/predict', async (req, res) => {
  try {
    const { homeTeam, awayTeam } = req.body;
    const prediction = await runPrediction({ homeTeam, awayTeam });
    res.json(prediction);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;