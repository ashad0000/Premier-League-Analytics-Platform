const { spawn } = require('child_process');
const path = require('path');

const runPrediction = (matchData) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../MLModelTraining/predict_matches.py'),
      JSON.stringify(matchData)
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(JSON.parse(output));
      } else {
        reject(new Error('Prediction failed'));
      }
    });
  });
};

module.exports = { runPrediction };