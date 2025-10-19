import React, { useState, useEffect, useRef } from 'react';
import { KeystrokeCollector } from './keystroke';
import './Widget.css';

/**
 * BiometricTextWidget - Main component for text-based biometric authentication
 *
 * Features:
 * - Paste prevention
 * - Keystroke timing collection
 * - Auto-submit on timeout
 * - Integration with backend API
 */
export default function BiometricTextWidget({ mode = 'enroll', userId, apiBase = '/api', onComplete }) {
  const [text, setText] = useState('');
  const [timeLeft, setTimeLeft] = useState(180); // 3 minutes default
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [challenge, setChallenge] = useState(null);
  const [allChallenges, setAllChallenges] = useState([]); // Store all challenges
  const [currentChallengeIndex, setCurrentChallengeIndex] = useState(0); // Track which challenge we're on
  const [sessionToken, setSessionToken] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [samplesRemaining, setSamplesRemaining] = useState(8);
  const [profileReady, setProfileReady] = useState(false);

  const textareaRef = useRef(null);
  const keystrokeCollector = useRef(new KeystrokeCollector());
  const timerRef = useRef(null);
  const pasteAttempts = useRef(0);

  // Initialize enrollment or verification
  useEffect(() => {
    if (mode === 'enroll') {
      startEnrollment();
    } else if (mode === 'verify') {
      // For verify mode, could prepare challenge
      setMessage('Please write naturally about your day or current thoughts...');
      setTimeLeft(180);
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [mode]);

  // Timer countdown
  useEffect(() => {
    if (timeLeft > 0) {
      timerRef.current = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            handleAutoSubmit();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [timeLeft, text]);

  /**
   * Start enrollment flow
   */
  const startEnrollment = async () => {
    try {
      const response = await fetch(`${apiBase}/enroll/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          lang: 'en',
          domain: 'chat'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setAllChallenges(data.challenges); // Store all challenges
      setChallenge(data.challenges[0]);
      setCurrentChallengeIndex(0);
      setSessionToken(data.session_token);
      setSamplesRemaining(data.required_samples);
      setMessage(data.challenges[0].prompt);
      setTimeLeft(data.challenges[0].timebox_s);
    } catch (err) {
      setError(`Failed to start enrollment: ${err.message}`);
    }
  };

  /**
   * Handle text input
   */
  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  /**
   * Prevent paste and log attempt
   */
  const handlePaste = (e) => {
    //Isko uncomment karo
    // e.preventDefault();
    // pasteAttempts.current += 1;
    // setError(`Paste not allowed! Please type naturally. (Attempt ${pasteAttempts.current})`);
    // setTimeout(() => setError(''), 3000);
  };

  /**
   * Track keystroke events
   */
  const handleKeyDown = (e) => {
    keystrokeCollector.current.recordEvent(e.key, 'down');
  };

  const handleKeyUp = (e) => {
    keystrokeCollector.current.recordEvent(e.key, 'up');
  };

  /**
   * Submit text sample
   */
  const handleSubmit = async () => {
    if (isSubmitting) return;

    const wordCount = text.trim().split(/\s+/).filter(w => w.length > 0).length;
    const minWords = challenge?.min_words || 50;

    if (wordCount < minWords) {
      setError(`Please write at least ${minWords} words. Current: ${wordCount}`);
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const timings = keystrokeCollector.current.getHistogram();

      let endpoint, body;
      if (mode === 'enroll') {
        endpoint = `${apiBase}/enroll/submit`;
        body = {
          user_id: userId,
          challenge_id: challenge.challenge_id,
          text: text,
          timings: timings,
          session_token: sessionToken
        };
      } else {
        endpoint = `${apiBase}/verify`;
        body = {
          user_id: userId,
          text: text,
          timings: timings,
          lang: 'en',
          domain_hint: 'chat'
        };
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();

      if (mode === 'enroll') {
        if (result.accepted) {
          setSamplesRemaining(result.remaining);

          if (result.profile_ready) {
            setProfileReady(true);
            setMessage('✅ Enrollment complete! Your profile is ready.');
            // Call onComplete callback if provided
            if (onComplete) {
              setTimeout(() => onComplete(), 2000);
            }
          } else {
            setMessage(`Sample ${8 - result.remaining} of 8 accepted. ${result.remaining} more to go!`);
            // Reset for next sample
            setText('');
            keystrokeCollector.current.reset();

            // Move to next challenge
            const nextIndex = currentChallengeIndex + 1;
            if (nextIndex < allChallenges.length) {
              setCurrentChallengeIndex(nextIndex);
              setChallenge(allChallenges[nextIndex]);
              setMessage(allChallenges[nextIndex].prompt);
              setTimeLeft(allChallenges[nextIndex].timebox_s);
            } else {
              // Fallback: reuse last challenge if we somehow run out
              setTimeLeft(challenge.timebox_s);
            }
          }
        } else {
          // Check for LLM detection warning
          const warnings = result.warnings || [];
          const isLLMDetected = warnings.some(w => w.includes('LLM_GENERATED_TEXT_DETECTED') || w.includes('AI-generated'));

          if (isLLMDetected) {
            setError('⚠️ AI-generated text detected! Please write naturally in your own words, not using AI tools like ChatGPT.');
          } else {
            setError(`Sample rejected: ${warnings.join(', ')}`);
          }

          // Keep text so user can edit and resubmit
          setIsSubmitting(false);
          return;
        }
      } else {
        // Verify mode
        setMessage(`Decision: ${result.decision.toUpperCase()} (Score: ${result.score.toFixed(2)})`);
        if (result.reasons.length > 0) {
          setMessage(msg => msg + ` - ${result.reasons.join(', ')}`);
        }
      }
    } catch (err) {
      setError(`Submission failed: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  /**
   * Auto-submit when timer expires
   */
  const handleAutoSubmit = () => {
    if (text.trim().length > 0 && !profileReady) {
      handleSubmit();
    }
  };

  /**
   * Format time for display
   */
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  /**
   * Get word count
   */
  const getWordCount = () => {
    return text.trim().split(/\s+/).filter(w => w.length > 0).length;
  };

  /**
   * Get timer bar color based on time remaining
   */
  const getTimerColor = () => {
    const percent = (timeLeft / (challenge?.timebox_s || 180)) * 100;
    if (percent > 50) return '#4CAF50';
    if (percent > 25) return '#FF9800';
    return '#F44336';
  };

  if (profileReady) {
    return (
      <div className="widget-container">
        <div className="success-message">
          <h2>✅ Enrollment Complete!</h2>
          <p>Your biometric text profile has been successfully created.</p>
          <p>You can now use text-based authentication.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="widget-container">
      <div className="widget-header">
        <h2>{mode === 'enroll' ? 'Text Enrollment' : 'Text Verification'}</h2>
        {mode === 'enroll' && (
          <div className="progress-indicator">
            Sample {8 - samplesRemaining + 1} of 8
          </div>
        )}
      </div>

      {message && (
        <div className="prompt-box">
          {message}
        </div>
      )}

      {error && (
        <div className="error-box">
          ⚠️ {error}
        </div>
      )}

      <div className="timer-container">
        <div className="timer-bar" style={{
          width: `${(timeLeft / (challenge?.timebox_s || 180)) * 100}%`,
          backgroundColor: getTimerColor()
        }} />
        <div className="timer-text">
          Time remaining: {formatTime(timeLeft)}
        </div>
      </div>

      <textarea
        ref={textareaRef}
        className="text-input"
        value={text}
        onChange={handleTextChange}
        onPaste={handlePaste}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        placeholder="Start typing naturally here..."
        disabled={isSubmitting || profileReady}
        rows={10}
      />

      <div className="stats-bar">
        <span>Words: {getWordCount()} / {challenge?.min_words || 50}</span>
        <span>Keystrokes: {keystrokeCollector.current.getEventCount()}</span>
        <span>Paste attempts: {pasteAttempts.current}</span>
      </div>

      <button
        className="submit-button"
        onClick={handleSubmit}
        disabled={isSubmitting || profileReady || getWordCount() < (challenge?.min_words || 50)}
      >
        {isSubmitting ? 'Submitting...' : 'Submit'}
      </button>

      <div className="info-text">
        ⓘ Paste is disabled. Please type naturally for accurate biometric analysis.
      </div>
    </div>
  );
}
