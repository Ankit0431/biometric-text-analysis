import React, { useState } from 'react';
import './Auth.css';

export default function AuthPage({ onLogin }) {
    const [isSignup, setIsSignup] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        username: '',
        password: ''
    });
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
        setError('');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setMessage('');
        setIsLoading(true);

        try {
            const endpoint = isSignup ? '/api/auth/signup' : '/api/auth/login';
            const body = isSignup
                ? formData
                : { username: formData.username, password: formData.password };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Authentication failed');
            }

            if (isSignup) {
                setMessage('Account created successfully! Please login.');
                setIsSignup(false);
                setFormData({ name: '', username: '', password: '' });
            } else {
                // Login successful
                onLogin({
                    userId: data.user_id,
                    username: data.username,
                    name: data.name,
                    biometricEnrolled: data.biometric_enrolled,
                    requiresMFA: data.requires_mfa
                });
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-box">
                <h1>{isSignup ? 'Sign Up' : 'Login'}</h1>
                <p className="auth-subtitle">
                    {isSignup
                        ? 'Create an account with biometric text authentication'
                        : 'Sign in to your account'}
                </p>

                {error && <div className="error-message">{error}</div>}
                {message && <div className="success-message">{message}</div>}

                <form onSubmit={handleSubmit}>
                    {isSignup && (
                        <div className="form-group">
                            <label htmlFor="name">Full Name</label>
                            <input
                                type="text"
                                id="name"
                                name="name"
                                value={formData.name}
                                onChange={handleChange}
                                required
                                placeholder="Enter your full name"
                            />
                        </div>
                    )}

                    <div className="form-group">
                        <label htmlFor="username">Username</label>
                        <input
                            type="text"
                            id="username"
                            name="username"
                            value={formData.username}
                            onChange={handleChange}
                            required
                            minLength="3"
                            placeholder="Enter your username"
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="password">Password</label>
                        <input
                            type="password"
                            id="password"
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            required
                            minLength="6"
                            placeholder="Enter your password"
                        />
                    </div>

                    <button
                        type="submit"
                        className="btn-primary"
                        disabled={isLoading}
                    >
                        {isLoading ? 'Please wait...' : (isSignup ? 'Sign Up' : 'Login')}
                    </button>
                </form>

                <div className="auth-toggle">
                    {isSignup ? (
                        <p>
                            Already have an account?{' '}
                            <button onClick={() => setIsSignup(false)} className="link-button">
                                Login here
                            </button>
                        </p>
                    ) : (
                        <p>
                            Don't have an account?{' '}
                            <button onClick={() => setIsSignup(true)} className="link-button">
                                Sign up here
                            </button>
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
}
