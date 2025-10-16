import React from 'react';
import BiometricTextWidget from './Widget';
import './App.css';

function App() {
  // For demo purposes - in production, userId would come from authentication
  const userId = 'demo_user_123';

  // Toggle between 'enroll' and 'verify' modes
  const [mode, setMode] = React.useState('enroll');

  return (
    <div className="App">
      <header className="App-header">
        <h1>Biometric Text Authentication Demo</h1>
        <div className="mode-selector">
          <button
            className={mode === 'enroll' ? 'active' : ''}
            onClick={() => setMode('enroll')}
          >
            Enrollment
          </button>
          <button
            className={mode === 'verify' ? 'active' : ''}
            onClick={() => setMode('verify')}
          >
            Verification
          </button>
        </div>
      </header>

      <main>
        <BiometricTextWidget
          mode={mode}
          userId={userId}
          apiBase="/api"
        />
      </main>

      <footer className="App-footer">
        <p>Biometric Text Authentication - Privacy-preserving typing biometrics</p>
      </footer>
    </div>
  );
}

export default App;
