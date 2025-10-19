import React from 'react';
import './Dashboard.css';

export default function Dashboard({ user, onLogout }) {
    return (
        <div className="dashboard-container">
            <div className="dashboard-header">
                <h1>Welcome, {user.name}! 👋</h1>
                <button className="btn-logout" onClick={onLogout}>
                    Logout
                </button>
            </div>

            <div className="dashboard-content">
                <div className="info-card">
                    <h2>Account Information</h2>
                    <div className="info-row">
                        <span className="info-label">Name:</span>
                        <span className="info-value">{user.name}</span>
                    </div>
                    <div className="info-row">
                        <span className="info-label">Username:</span>
                        <span className="info-value">{user.username}</span>
                    </div>
                    <div className="info-row">
                        <span className="info-label">User ID:</span>
                        <span className="info-value">{user.userId}</span>
                    </div>
                    <div className="info-row">
                        <span className="info-label">Biometric Status:</span>
                        <span className={`badge ${user.biometricEnrolled ? 'badge-success' : 'badge-warning'}`}>
                            {user.biometricEnrolled ? '✓ Enrolled' : '⚠ Not Enrolled'}
                        </span>
                    </div>
                </div>

                <div className="welcome-card">
                    <h2>🎉 Authentication Successful!</h2>
                    <p>
                        You have successfully authenticated using our advanced biometric text analysis system.
                        Your typing patterns and writing style were verified to confirm your identity.
                    </p>
                    <div className="security-info">
                        <h3>🔒 Security Features:</h3>
                        <ul>
                            <li>Two-factor authentication with biometric text analysis</li>
                            <li>1:1 matching ensures only your profile is used for verification</li>
                            <li>Privacy-preserving: Raw text is never stored</li>
                            <li>Adaptive thresholds based on your writing patterns</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
}
