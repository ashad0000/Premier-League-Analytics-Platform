const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

// Import routes
const teamRoutes = require('./routes/teams');
const playerRoutes = require('./routes/players');
const fixtureRoutes = require('./routes/fixtures');
const predictionsRoutes = require('./routes/predictions');

// Use routes
app.use('/api/teams', teamRoutes);
app.use('/api/players', playerRoutes);
app.use('/api/fixtures', fixtureRoutes);
app.use('/api/predictions', predictionsRoutes);

app.get('/', (req, res) => {
  res.send('Soccer App Backend is running ⚽');
});

// Only start server if NOT running tests
if (process.env.NODE_ENV !== 'test') {
  const PORT = process.env.PORT || 5000;
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

// Export the Express app for testing
module.exports = app;
