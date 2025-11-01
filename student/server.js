const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 8080;

// Serve static files from build/web
app.use(express.static(path.join(__dirname, 'build/web')));

// Handle all routes - return index.html for client-side routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build/web', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Flutter web app running on port ${PORT}`);
});