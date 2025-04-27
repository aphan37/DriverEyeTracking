import React from 'react';

function EyeStatus({ classification }) {
  return (
    <div style={{ padding: 20 }}>
      <h2>Eye Status: {classification}</h2>
    </div>
  );
}

export default EyeStatus;
