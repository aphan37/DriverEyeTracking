import React from 'react';

function SleepinessGauge({ score }) {
  return (
    <div style={{ padding: 20 }}>
      <h2>Sleepiness Score: {score}%</h2>
      <div style={{
        width: '100%',
        height: '30px',
        backgroundColor: '#ddd',
        marginTop: '10px'
      }}>
        <div style={{
          width: `${score}%`,
          height: '100%',
          backgroundColor: score > 50 ? 'green' : (score > 30 ? 'orange' : 'red')
        }} />
      </div>
    </div>
  );
}

export default SleepinessGauge;
