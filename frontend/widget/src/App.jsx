import React, { useState } from 'react';
import BiometricTextWidget from './Widget';
import AuthPage from './AuthPage';
import BiometricMFA from './BiometricMFA';
import Dashboard from './Dashboard';
import GuessUser from './GuessUser';
import './App.css';

function App() {
  const [authState, setAuthState] = useState('login'); // 'login', 'enroll', 'mfa', 'dashboard', 'guess-user'
  const [user, setUser] = useState(null);

  const handleLogin = (userData) => {
    setUser(userData);
    if (!userData.biometricEnrolled) {
      // User needs to enroll first
      setAuthState('enroll');
    } else if (userData.requiresMFA) {
      // User needs to complete MFA
      setAuthState('mfa');
    } else {
      // Direct to dashboard (shouldn't happen with current logic)
      setAuthState('dashboard');
    }
  };

  const handleEnrollmentComplete = () => {
    // After enrollment, go to MFA for first login
    setAuthState('mfa');
  };

  const handleMFASuccess = () => {
    // MFA successful, go to dashboard
    setAuthState('dashboard');
  };

  const handleMFACancel = () => {
    // User cancelled MFA, back to login
    setUser(null);
    setAuthState('login');
  };

  const handleLogout = () => {
    setUser(null);
    setAuthState('login');
  };

  const handleGuessUser = () => {
    console.log('GuessUser button clicked!');
    setAuthState('guess-user');
  };

  const handleBackToLogin = () => {
    setAuthState('login');
  };

  return (
    <div className="App">
      {authState === 'login' && (
        <AuthPage onLogin={handleLogin} onGuessUser={handleGuessUser} />
      )}

      {authState === 'guess-user' && (
        <GuessUser onBack={handleBackToLogin} />
      )}

      {authState === 'enroll' && user && (
        <>
          <header className="App-header">
            <h1>Biometric Enrollment</h1>
            <p>Welcome, {user.name}! Please complete your biometric enrollment.</p>
          </header>
          <main>
            <BiometricTextWidget
              mode="enroll"
              userId={user.userId}
              apiBase="/api"
              onComplete={handleEnrollmentComplete}
            />
          </main>
        </>
      )}

      {authState === 'mfa' && user && (
        <BiometricMFA
          userId={user.userId}
          username={user.username}
          onSuccess={handleMFASuccess}
          onCancel={handleMFACancel}
        />
      )}

      {authState === 'dashboard' && user && (
        <Dashboard
          user={user}
          onLogout={handleLogout}
        />
      )}
    </div>
  );
}

export default App;
