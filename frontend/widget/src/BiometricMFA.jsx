import React, { useState, useRef, useEffect } from 'react';
import { KeystrokeCollector } from './keystroke';
import './BiometricMFA.css';

export default function BiometricMFA({ userId, username, onSuccess, onCancel }) {
    const [text, setText] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');
    const [challenge, setChallenge] = useState(null);
    const [isLoadingChallenge, setIsLoadingChallenge] = useState(true);

    const keystrokeCollector = useRef(new KeystrokeCollector());
    const pasteAttempts = useRef(0);

    // Fetch MFA challenge on component mount
    useEffect(() => {
        fetchMFAChallenge();
    }, [userId]);

    const fetchMFAChallenge = async () => {
        setIsLoadingChallenge(true);
        setError('');

        try {
            const response = await fetch(`/api/auth/mfa-challenge?user_id=${userId}`);

            if (!response.ok) {
                const result = await response.json();
                throw new Error(result.detail || 'Failed to load challenge');
            }

            const challengeData = await response.json();
            setChallenge(challengeData);
        } catch (err) {
            setError(err.message || 'Failed to load verification challenge');
        } finally {
            setIsLoadingChallenge(false);
        }
    };

    const handleTextChange = (e) => {
        const newText = e.target.value;

        // Detect paste
        if (newText.length - text.length > 10) {
            pasteAttempts.current += 1;
            if (pasteAttempts.current >= 3) {
                setError('Pasting is not allowed. Please type naturally.');
                return;
            }
            setMessage(`Warning: Pasting detected (${pasteAttempts.current}/3). Please type naturally.`);
            setTimeout(() => setMessage(''), 3000);
        }

        setText(newText);
    };

    const handleKeyDown = (e) => {
        keystrokeCollector.current.recordEvent(e.key, 'down');
    };

    const handleKeyUp = (e) => {
        keystrokeCollector.current.recordEvent(e.key, 'up');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (text.trim().split(/\s+/).length < (challenge?.min_words || 50)) {
            setError(`Please write at least ${challenge?.min_words || 50} words for accurate verification.`);
            return;
        }

        setIsSubmitting(true);
        setError('');
        setMessage('');

        try {
            const timings = keystrokeCollector.current.getHistogram();

            const response = await fetch('/api/auth/biometric-mfa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    text: text,
                    challenge_id: challenge?.challenge_id,
                    timings: timings
                })
            });

            const result = await response.json();

            if (!response.ok) {
                // Handle validation errors specifically
                if (response.status === 422 && result.detail) {
                    if (Array.isArray(result.detail)) {
                        const errors = result.detail.map(err => err.msg).join(', ');
                        throw new Error(`Validation error: ${errors}`);
                    }
                    throw new Error(result.detail);
                }

                // Handle LLM detection
                if (result.message && result.message.includes('AI-generated')) {
                    throw new Error('⚠️ AI-generated text detected! Please write naturally in your own words, not using AI tools like ChatGPT.');
                }

                throw new Error(result.detail || result.message || 'Verification failed');
            }

            if (result.success) {
                onSuccess();
            } else {
                // Check for LLM detection in reasons
                const isLLMDetected = result.reasons && result.reasons.some(r => r.includes('LLM_GENERATED'));

                if (isLLMDetected) {
                    setError('⚠️ AI-generated text detected! Please write naturally in your own words, not using AI tools like ChatGPT.');
                } else {
                    setError(result.message || 'Authentication failed. Your text does not match your enrolled pattern.');
                }
            }
        } catch (err) {
            setError(err.message || 'An error occurred during verification');
        } finally {
            setIsSubmitting(false);
        }
    };

    const wordCount = text.trim().split(/\s+/).filter(w => w.length > 0).length;
    const minWords = challenge?.min_words || 50;

    if (isLoadingChallenge) {
        return (
            <div className="mfa-container">
                <div className="mfa-box">
                    <div className="mfa-header">
                        <h1>🔐 Loading...</h1>
                        <p>Preparing your verification challenge...</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="mfa-container">
            <div className="mfa-box">
                <div className="mfa-header">
                    <h1>🔐 Biometric Text Verification</h1>
                    <p className="mfa-subtitle">
                        Welcome back, <strong>{username}</strong>!
                    </p>
                    <div className="challenge-prompt">
                        <p className="mfa-instruction">
                            <strong>Please respond to the following prompt:</strong>
                        </p>
                        <div className="prompt-text">
                            "{challenge?.prompt || 'Write naturally about any topic...'}"
                        </div>
                        <p className="mfa-instruction-sub">
                            Write at least {minWords} words. Your typing pattern and writing style will be verified.
                        </p>
                    </div>
                </div>

                {error && <div className="error-message">{error}</div>}
                {message && <div className="warning-message">{message}</div>}

                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="text">
                            Your response (no pasting allowed)
                        </label>
                        <textarea
                            id="text"
                            value={text}
                            onChange={handleTextChange}
                            onKeyDown={handleKeyDown}
                            onKeyUp={handleKeyUp}
                            placeholder="Write your response here..."
                            rows="10"
                            disabled={isSubmitting}
                            autoFocus
                        />
                        <div className="word-count">
                            {wordCount} / {minWords} words
                            {wordCount >= minWords && ' ✓'}
                        </div>
                    </div>

                    <div className="button-group">
                        <button
                            type="button"
                            onClick={onCancel}
                            className="btn-secondary"
                            disabled={isSubmitting}
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="btn-primary"
                            disabled={isSubmitting || wordCount < minWords}
                        >
                            {isSubmitting ? 'Verifying...' : 'Verify Identity'}
                        </button>
                    </div>
                </form>

                <div className="mfa-info">
                    <p>
                        <strong>Note:</strong> This is a 1:1 biometric verification.
                        Only your enrolled profile is used for matching.
                    </p>
                </div>
            </div>
        </div>
    );
}
