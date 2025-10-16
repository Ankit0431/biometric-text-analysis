/**
 * Keystroke timing utilities for biometric text analysis.
 *
 * Collects keystroke timing data (inter-key intervals) and converts
 * to binned histograms for privacy-preserving transmission to server.
 */

/**
 * Keystroke collector that tracks timing data.
 */
export class KeystrokeCollector {
  constructor() {
    this.events = [];
    this.lastTimestamp = null;
    this.ikis = []; // Inter-key intervals
  }

  /**
   * Record a keystroke event.
   * @param {string} key - The key pressed
   * @param {'down'|'up'} type - Key event type
   */
  recordEvent(key, type) {
    const timestamp = Date.now();

    this.events.push({ key, timestamp, type });

    // Calculate inter-key interval (time between consecutive key presses)
    if (type === 'down' && this.lastTimestamp !== null) {
      const iki = timestamp - this.lastTimestamp;
      if (iki > 0 && iki < 5000) { // Ignore pauses > 5 seconds
        this.ikis.push(iki);
      }
    }

    if (type === 'down') {
      this.lastTimestamp = timestamp;
    }
  }

  /**
   * Convert timing data to binned histogram for privacy.
   * Bins: [0-50ms, 50-100ms, 100-150ms, 150-200ms, 200-300ms, 300+ms]
   * @returns {Object} Histogram data with statistics
   */
  getHistogram() {
    if (this.ikis.length === 0) {
      return {
        histogram: [0, 0, 0, 0, 0, 0],
        mean_iki: 0,
        std_iki: 0,
        total_events: 0,
      };
    }

    // Define bins (in milliseconds)
    const bins = [50, 100, 150, 200, 300, Infinity];
    const histogram = new Array(bins.length).fill(0);

    // Bin the IKIs
    for (const iki of this.ikis) {
      for (let i = 0; i < bins.length; i++) {
        if (iki < bins[i]) {
          histogram[i]++;
          break;
        }
      }
    }

    // Calculate statistics
    const mean = this.ikis.reduce((sum, iki) => sum + iki, 0) / this.ikis.length;
    const variance = this.ikis.reduce((sum, iki) => sum + Math.pow(iki - mean, 2), 0) / this.ikis.length;
    const std = Math.sqrt(variance);

    return {
      histogram,
      mean_iki: Math.round(mean),
      std_iki: Math.round(std),
      total_events: this.ikis.length,
    };
  }

  /**
   * Reset collected data.
   */
  reset() {
    this.events = [];
    this.ikis = [];
    this.lastTimestamp = null;
  }

  /**
   * Get raw event count (for testing).
   * @returns {number} Number of recorded events
   */
  getEventCount() {
    return this.events.length;
  }
}
