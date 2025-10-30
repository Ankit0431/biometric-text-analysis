import React, { useState, useRef } from 'react';
import { KeystrokeCollector } from './keystroke';
import './GuessUser.css';

const GuessUser = ({ onBack }) => {
  const [text, setText] = useState('');
  const [events, setEvents] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const keystrokeCollector = useRef(new KeystrokeCollector());

  const handleKeyDown = (e) => {
    keystrokeCollector.current.recordEvent(e.key, 'down');
    // Update events state for display purposes
    setEvents(prev => [...prev, { key: e.key, timestamp: Date.now(), type: 'keydown' }]);
  };

  const handleKeyUp = (e) => {
    keystrokeCollector.current.recordEvent(e.key, 'up');
  };

  const handleAnalyze = async () => {
    if (text.trim().length < 10) {
      setError('Please enter at least 10 characters for analysis');
      return;
    }

    setIsAnalyzing(true);
    setError('');
    setResult(null);

    try {
      // Get histogram data from keystroke collector
      const histogramData = keystrokeCollector.current.getHistogram();
      
      // Debug: log what we're sending
      console.log('Sending keystroke data:', histogramData);
      
      const response = await fetch('/api/auth/identify-user', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text.trim(),
          timings: histogramData.total_events > 0 ? histogramData : null,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError('Failed to analyze text. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClear = () => {
    setText('');
    setEvents([]);
    setResult(null);
    setError('');
    keystrokeCollector.current.reset();
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#22c55e'; // green
    if (confidence >= 0.6) return '#f59e0b'; // amber
    return '#ef4444'; // red
  };

  const formatConfidence = (confidence) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  return (
    <div className="guess-user-container">
      <div className="guess-user-header">
        <h1>🔍 Guess the User</h1>
        <p>Type some text and we'll try to identify who you are based on your writing patterns!</p>
        <button className="back-button" onClick={onBack}>
          ← Back to Login
        </button>
      </div>

      <div className="guess-user-content">
        <div className="input-section">
          <label htmlFor="text-input">Enter your text:</label>
          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            onKeyUp={handleKeyUp}
            placeholder="Type naturally... The more you write, the more accurate the identification will be. Try to write at least 2-3 sentences about any topic."
            rows={6}
            disabled={isAnalyzing}
          />
          
          <div className="input-stats">
            <span>Characters: {text.length}</span>
            <span>Keystrokes captured: {keystrokeCollector.current.getEventCount()}</span>
          </div>

          <div className="action-buttons">
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={isAnalyzing || text.trim().length < 10}
            >
              {isAnalyzing ? '🔄 Analyzing...' : '🔍 Guess User'}
            </button>
            <button
              className="clear-button"
              onClick={handleClear}
              disabled={isAnalyzing}
            >
              Clear
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}
        </div>

        {result && (
          <div className="result-section">
            <h2>🎯 Identification Result</h2>
            
            <div className="main-result">
              {result.identified_user ? (
                <div className="identified">
                  <div className="user-info">
                    <h3>✅ Identified as: <strong>{result.username}</strong></h3>
                    <div 
                      className="confidence-score"
                      style={{ color: getConfidenceColor(result.confidence_score) }}
                    >
                      Confidence: {formatConfidence(result.confidence_score)}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="unknown">
                  <h3>❓ Unknown User</h3>
                  <div className="confidence-score">
                    Best match confidence: {formatConfidence(result.confidence_score)}
                  </div>
                </div>
              )}
              
              <div className="result-message">
                {result.message}
              </div>
            </div>

            {Object.keys(result.all_scores).length > 0 && (
              <div className="all-scores">
                <h4>📊 Scores for all enrolled users:</h4>
                <div className="scores-grid">
                  {Object.entries(result.all_scores)
                    .sort(([,a], [,b]) => b - a)
                    .map(([userId, score]) => (
                      <div 
                        key={userId} 
                        className={`score-item ${result.identified_user === userId ? 'best-match' : ''}`}
                      >
                        <span className="user-id">{userId}</span>
                        <div className="score-bar">
                          <div 
                            className="score-fill"
                            style={{ 
                              width: `${score * 100}%`,
                              backgroundColor: result.identified_user === userId ? '#22c55e' : '#e5e7eb'
                            }}
                          />
                          <span className="score-text">{score.toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            <div className="result-actions">
              <button className="try-again-button" onClick={handleClear}>
                🔄 Try Again
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="guess-user-info">
        <h3>ℹ️ How it works</h3>
        <ul>
          <li><strong>Text Analysis:</strong> We analyze your writing style, vocabulary, and sentence patterns</li>
          <li><strong>Keystroke Patterns:</strong> If available, we also analyze your typing rhythm and timing</li>
          <li><strong>1:N Matching:</strong> Your text is compared against all enrolled users to find the best match</li>
          <li><strong>Confidence Threshold:</strong> Only matches above 65% confidence are identified</li>
        </ul>
        
        <div className="privacy-note">
          <strong>🔒 Privacy:</strong> Your text is only used for analysis and is not stored permanently.
        </div>
      </div>
    </div>
  );
};

export default GuessUser;