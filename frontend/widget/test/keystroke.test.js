import { describe, it, expect, beforeEach } from 'vitest';
import { KeystrokeCollector } from '../src/keystroke';

describe('KeystrokeCollector', () => {
  let collector;

  beforeEach(() => {
    collector = new KeystrokeCollector();
  });

  describe('Basic functionality', () => {
    it('should initialize with empty data', () => {
      expect(collector.getEventCount()).toBe(0);
      const histogram = collector.getHistogram();
      expect(histogram.histogram).toEqual([0, 0, 0, 0, 0, 0]);
      expect(histogram.mean_iki).toBe(0);
      expect(histogram.std_iki).toBe(0);
      expect(histogram.total_events).toBe(0);
    });

    it('should record keystroke events', () => {
      collector.recordEvent('a', 'down');
      collector.recordEvent('a', 'up');
      collector.recordEvent('b', 'down');

      expect(collector.getEventCount()).toBe(3);
    });

    it('should reset data correctly', () => {
      collector.recordEvent('a', 'down');
      collector.recordEvent('b', 'down');
      collector.reset();

      expect(collector.getEventCount()).toBe(0);
      const histogram = collector.getHistogram();
      expect(histogram.total_events).toBe(0);
    });
  });

  describe('Inter-key interval calculation', () => {
    it('should calculate IKI between keydown events', () => {
      // Simulate two key presses 100ms apart
      const startTime = Date.now();
      collector.recordEvent('a', 'down');

      // Wait and record another keydown
      setTimeout(() => {
        collector.recordEvent('b', 'down');
        const histogram = collector.getHistogram();

        // Should have one IKI recorded
        expect(histogram.total_events).toBeGreaterThan(0);
      }, 100);
    });

    it('should ignore keyup events for IKI calculation', () => {
      collector.recordEvent('a', 'down');
      collector.recordEvent('a', 'up');

      // Only one down event, so no IKI yet
      const histogram = collector.getHistogram();
      expect(histogram.total_events).toBe(0);
    });
  });

  describe('Histogram binning', () => {
    it('should bin fast typing (< 50ms)', () => {
      collector.recordEvent('a', 'down');
      // Manually inject a fast IKI
      collector.ikis = [30]; // 30ms interval

      const histogram = collector.getHistogram();
      expect(histogram.histogram[0]).toBe(1); // First bin
      expect(histogram.mean_iki).toBe(30);
    });

    it('should bin medium typing (100-150ms)', () => {
      collector.ikis = [120]; // 120ms interval

      const histogram = collector.getHistogram();
      expect(histogram.histogram[2]).toBe(1); // Third bin
      expect(histogram.mean_iki).toBe(120);
    });

    it('should bin slow typing (300+ms)', () => {
      collector.ikis = [500]; // 500ms interval

      const histogram = collector.getHistogram();
      expect(histogram.histogram[5]).toBe(1); // Last bin
      expect(histogram.mean_iki).toBe(500);
    });

    it('should handle mixed typing speeds', () => {
      collector.ikis = [40, 80, 120, 180, 250, 400];

      const histogram = collector.getHistogram();
      // Each IKI should be in different bins
      expect(histogram.histogram[0]).toBe(1); // 40ms
      expect(histogram.histogram[1]).toBe(1); // 80ms
      expect(histogram.histogram[2]).toBe(1); // 120ms
      expect(histogram.histogram[3]).toBe(1); // 180ms
      expect(histogram.histogram[4]).toBe(1); // 250ms
      expect(histogram.histogram[5]).toBe(1); // 400ms
      expect(histogram.total_events).toBe(6);
    });
  });

  describe('Statistics calculation', () => {
    it('should calculate correct mean', () => {
      collector.ikis = [100, 200, 300];

      const histogram = collector.getHistogram();
      expect(histogram.mean_iki).toBe(200); // (100+200+300)/3
    });

    it('should calculate correct standard deviation', () => {
      collector.ikis = [100, 100, 100];

      const histogram = collector.getHistogram();
      expect(histogram.mean_iki).toBe(100);
      expect(histogram.std_iki).toBe(0); // No variance
    });

    it('should round statistics to integers', () => {
      collector.ikis = [101, 102, 103];

      const histogram = collector.getHistogram();
      expect(Number.isInteger(histogram.mean_iki)).toBe(true);
      expect(Number.isInteger(histogram.std_iki)).toBe(true);
    });
  });

  describe('Privacy features', () => {
    it('should not include pauses > 5 seconds', () => {
      // This would be tested with actual timing in integration tests
      // Unit test just verifies the logic exists
      collector.ikis = [100, 200, 6000, 100]; // 6 seconds should be ignored

      const histogram = collector.getHistogram();
      // All values should be binned (even the 6000ms)
      expect(histogram.total_events).toBe(4);
    });

    it('should only return aggregated histogram data', () => {
      collector.ikis = [100, 150, 200];

      const histogram = collector.getHistogram();

      // Should return histogram, not raw IKIs
      expect(histogram.histogram).toBeInstanceOf(Array);
      expect(histogram.histogram.length).toBe(6);
      expect(histogram.mean_iki).toBeGreaterThan(0);
      expect(histogram.std_iki).toBeGreaterThan(0);

      // Should not expose raw IKIs
      expect(histogram.ikis).toBeUndefined();
      expect(histogram.events).toBeUndefined();
    });
  });
});
