import React, { useEffect, useState } from 'react';
import SleepinessGauge from './components/SleepinessGauge';
import EyeStatus from './components/EyeStatus';

function App() {
  const [score, setScore] = useState(100);
  const [classification, setClassification] = useState("Open");

  useEffect(() => {
    const interval = setInterval(() => {
      fetch('http://localhost:5000/api/sleepiness')
        .then(res => res.json())
        .then(data => {
          setScore(data.sleepiness_score);
          setClassification(data.classification);
        })
        .catch(err => console.error(err));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ textAlign: 'center', marginTop: 50 }}>
      <h1>Driver Drowsiness Dashboard</h1>
      <SleepinessGauge score={score} />
      <EyeStatus classification={classification} />
    </div>
  );
}

export default App;
