// frontend/src/App.js
import React, { useEffect, useState } from 'react';
import Sidebar from './components/sidebar';
import Home from './pages/Home';

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    // Fetch data from your Express backend
    fetch('http://localhost:5000/')
      .then(response => response.text())
      .then(data => setMessage(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <div style={{ display: "flex" }}>
      <Sidebar active="Home" />
      <Home />
    </div>
  );
}

export default App;
