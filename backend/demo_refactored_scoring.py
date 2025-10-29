#!/usr/bin/env python3
"""
Demo script showcasing the refactored scoring pipeline with hybrid feature fusion.

This demonstrates:
1. Individual component scores (semantic, stylometry, keystroke)
2. Adaptive weighting based on component availability
3. Score normalization and fusion
4. LLM penalty integration
5. Detailed logging for debugging
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from scoring import compute_final_score, score_sample


def demo_basic_fusion():
    """Demonstrate basic hybrid fusion scoring."""
    print("=== DEMO 1: Basic Hybrid Fusion ===")
    
    # High-quality sample (all components agree)
    semantic_score = 0.85
    stylometry_score = 0.80
    keystroke_score = 0.75
    llm_penalty = 0.1
    
    print(f"Input scores: semantic={semantic_score}, stylometry={stylometry_score}, keystroke={keystroke_score}")
    print(f"LLM penalty: {llm_penalty}")
    
    final_score = compute_final_score(semantic_score, stylometry_score, keystroke_score, llm_penalty)
    print(f"Final fused score: {final_score:.4f}")
    print()


def demo_adaptive_weighting():
    """Demonstrate adaptive weighting when keystroke data is unavailable."""
    print("=== DEMO 2: Adaptive Weighting (No Keystroke Data) ===")
    
    semantic_score = 0.85
    stylometry_score = 0.80
    keystroke_score = None  # Missing keystroke data
    llm_penalty = 0.1
    
    print(f"Input scores: semantic={semantic_score}, stylometry={stylometry_score}, keystroke=None")
    print(f"LLM penalty: {llm_penalty}")
    
    final_score = compute_final_score(semantic_score, stylometry_score, keystroke_score, llm_penalty)
    print(f"Final fused score (adaptive weights): {final_score:.4f}")
    print()


def demo_high_llm_penalty():
    """Demonstrate scoring with high LLM penalty."""
    print("=== DEMO 3: High LLM Penalty Detection ===")
    
    semantic_score = 0.85
    stylometry_score = 0.80
    keystroke_score = 0.75
    llm_penalty = 0.6  # High penalty (likely AI-generated)
    
    print(f"Input scores: semantic={semantic_score}, stylometry={stylometry_score}, keystroke={keystroke_score}")
    print(f"LLM penalty: {llm_penalty} (HIGH - likely AI-generated)")
    
    final_score = compute_final_score(semantic_score, stylometry_score, keystroke_score, llm_penalty)
    print(f"Final fused score (with penalty): {final_score:.4f}")
    print()


def demo_score_normalization():
    """Demonstrate score normalization behavior."""
    print("=== DEMO 4: Score Normalization ===")
    
    # Test case 1: Varied scores (normalization applied)
    print("Case 1: Varied scores (normalization applied)")
    final_score = compute_final_score(0.9, 0.3, 0.6, 0.0)
    print(f"Varied scores (0.9, 0.3, 0.6) -> {final_score:.4f}")
    print()
    
    # Test case 2: Similar scores (normalization skipped)
    print("Case 2: Similar scores (normalization skipped)")
    final_score = compute_final_score(0.8, 0.85, 0.82, 0.0)
    print(f"Similar scores (0.8, 0.85, 0.82) -> {final_score:.4f}")
    print()


def demo_copy_paste_detection():
    """Demonstrate copy-paste detection through keystroke analysis."""
    print("=== DEMO 5: Copy-Paste Detection ===")
    
    # Create mock profile
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    style = np.random.randn(100).astype(np.float32)
    
    profile = {
        'centroid': embedding.copy(),
        'style_mean': style.copy(), 
        'style_std': np.ones(100, dtype=np.float32) * 0.1,
        'keystroke_mean': np.random.rand(10).astype(np.float32),
    }
    
    text = "This is a long text message that should have many keystrokes if it was actually typed by the user."
    
    # Case 1: Normal typing (sufficient keystrokes)
    print("Case 1: Normal typing")
    timings_normal = {
        'total_events': len(text),  # 1 event per character
        'mean_iki': 150.0,
        'std_iki': 50.0,
        'histogram': [10, 20, 30, 25, 10, 5]
    }
    
    result = score_sample(profile, text, embedding, style, timings_normal)
    print(f"Normal typing - Keystroke score: {result['keystroke_score']:.4f}, Final: {result['final_score']:.4f}")
    
    # Case 2: Copy-paste (very few keystrokes)
    print("\nCase 2: Copy-paste detected")
    timings_paste = {
        'total_events': 5,  # Very few events for long text
        'mean_iki': 150.0,
        'std_iki': 50.0,
        'histogram': [1, 1, 1, 1, 1, 0]
    }
    
    result = score_sample(profile, text, embedding, style, timings_paste)
    print(f"Copy-paste - Keystroke score: {result['keystroke_score']:.4f}, Final: {result['final_score']:.4f}")
    print()


def demo_extreme_cases():
    """Demonstrate behavior with extreme input values."""
    print("=== DEMO 6: Extreme Cases ===")
    
    # Perfect scores
    print("Perfect scores (1.0, 1.0, 1.0):")
    final_score = compute_final_score(1.0, 1.0, 1.0, 0.0)
    print(f"Final score: {final_score:.4f}")
    print()
    
    # Worst scores
    print("Worst scores (0.0, 0.0, 0.0):")
    final_score = compute_final_score(0.0, 0.0, 0.0, 0.0)
    print(f"Final score: {final_score:.4f}")
    print()
    
    # Mixed with maximum penalty
    print("Mixed scores with maximum LLM penalty:")
    final_score = compute_final_score(0.8, 0.7, 0.6, 1.0)
    print(f"Final score: {final_score:.4f}")
    print()


if __name__ == "__main__":
    print("Biometric Text Analysis - Refactored Scoring Pipeline Demo")
    print("="*60)
    print()
    
    demo_basic_fusion()
    demo_adaptive_weighting() 
    demo_high_llm_penalty()
    demo_score_normalization()
    demo_copy_paste_detection()
    demo_extreme_cases()
    
    print("Demo completed! The refactored scoring pipeline includes:")
    print("✓ Hybrid feature fusion (semantic + stylometric + keystroke + LLM penalty)")
    print("✓ Proper score normalization with Z-score and min-max scaling")
    print("✓ Adaptive weighting based on component availability")
    print("✓ Enhanced LLM penalty weighting for high-confidence AI detection")
    print("✓ Detailed logging for debugging and transparency")
    print("✓ Copy-paste detection through keystroke ratio analysis")