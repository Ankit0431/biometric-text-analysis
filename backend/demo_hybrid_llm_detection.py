#!/usr/bin/env python3
"""
Demo script for the new Hybrid LLM Detection System.

This demonstrates:
1. Feature extraction from text samples
2. Heuristic-only detection (fast fallback)
3. Hybrid ML + heuristic detection (when available)
4. Comparison between detection methods
5. Integration with the scoring pipeline
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from llm_detection import (
    extract_llm_features,
    heuristic_llm_detection,
    detect_llm_likeness,
    initialize_llm_detection,
    SKLEARN_AVAILABLE,
    NLTK_AVAILABLE,
    TRANSFORMERS_AVAILABLE
)
from scoring import detect_llm_likeness as scoring_detect_llm


def demo_feature_extraction():
    """Demonstrate feature extraction from different text types."""
    print("=== DEMO 1: Feature Extraction ===")
    
    # Human-written text sample
    human_text = """
    Hey there! So I was thinking about our project today, and honestly, I'm not sure 
    we're on the right track. Like, the whole approach feels kinda off to me? 
    I mean, don't get me wrong - the basic idea is solid. But the implementation... 
    that's where I'm having doubts. What do you think? Should we maybe try a 
    different angle?
    """
    
    # AI-generated text sample (typical ChatGPT style)
    ai_text = """
    In consideration of the aforementioned project parameters, it is important to 
    note that the current approach presents several compelling advantages. Furthermore, 
    the implementation strategy demonstrates a comprehensive understanding of the 
    underlying principles. However, it is worth noting that alternative methodologies 
    might offer additional benefits. Consequently, a thorough evaluation of all 
    available options would be prudent before proceeding with the final implementation.
    """
    
    print("Human text features:")
    human_features = extract_llm_features(human_text)
    print(f"Feature vector shape: {human_features.shape}")
    print(f"Sample features: {human_features[:10]}")
    
    print("\nAI text features:")
    ai_features = extract_llm_features(ai_text)
    print(f"Feature vector shape: {ai_features.shape}")
    print(f"Sample features: {ai_features[:10]}")
    
    print(f"\nFeature differences (|AI - Human|):")
    diff = np.abs(ai_features - human_features)
    print(f"Mean absolute difference: {np.mean(diff):.4f}")
    print(f"Max difference: {np.max(diff):.4f}")
    print()


def demo_heuristic_detection():
    """Demonstrate heuristic-only detection."""
    print("=== DEMO 2: Heuristic Detection ===")
    
    test_samples = [
        ("Human casual", "Hey! How's it going? I was just wondering if you've seen the new movie. It's pretty good tbh, though the ending was kinda weird lol."),
        ("Human formal", "I am writing to inquire about the status of my application. I submitted all required documents last week and wanted to follow up on the next steps."),
        ("AI formal", "It is important to note that the implementation of this system requires careful consideration of multiple factors. Furthermore, the proposed solution demonstrates significant advantages over existing methodologies. Consequently, I would recommend proceeding with the outlined approach."),
        ("AI casual attempt", "I think this approach is really interesting. Additionally, it provides many benefits. Furthermore, the implementation seems straightforward. However, there are some challenges to consider."),
    ]
    
    for label, text in test_samples:
        penalty, is_llm = heuristic_llm_detection(text)
        status = "🤖 AI-like" if is_llm else "👤 Human-like"
        print(f"{label:15} | Penalty: {penalty:.3f} | {status}")
    
    print()


def demo_hybrid_detection():
    """Demonstrate hybrid ML + heuristic detection."""
    print("=== DEMO 3: Hybrid Detection ===")
    
    # Check system capabilities
    print(f"System capabilities:")
    print(f"  ✓ Heuristic detection: Always available")
    print(f"  {'✓' if SKLEARN_AVAILABLE else '✗'} ML classification: {'Available' if SKLEARN_AVAILABLE else 'Not available'}")
    print(f"  {'✓' if NLTK_AVAILABLE else '✗'} POS tagging: {'Available' if NLTK_AVAILABLE else 'Not available'}")
    print(f"  {'✓' if TRANSFORMERS_AVAILABLE else '✗'} Perplexity: {'Available' if TRANSFORMERS_AVAILABLE else 'Not available'}")
    print()
    
    test_samples = [
        ("Short human", "Thanks! Sounds good."),
        ("Human story", "So yesterday I went to the store and saw this crazy thing happen. This guy was trying to return a watermelon that he'd already eaten half of! The cashier just stared at him like he was nuts. I couldn't help but laugh - I mean, who does that?!"),
        ("AI summary", "The implementation of artificial intelligence systems requires comprehensive evaluation of multiple parameters. It is essential to consider the various factors that influence system performance. Furthermore, proper documentation and testing procedures must be established to ensure optimal functionality."),
        ("AI trying to be human", "I really think this is awesome! Additionally, it's super cool how everything works together. Moreover, the benefits are quite significant. However, we should also think about potential challenges."),
    ]
    
    for label, text in test_samples:
        try:
            penalty, is_llm = detect_llm_likeness(text, use_ml=True)
            status = "🤖 AI-like" if is_llm else "👤 Human-like"
            print(f"{label:15} | Penalty: {penalty:.3f} | {status}")
        except Exception as e:
            print(f"{label:15} | ERROR: {e}")
    
    print()


def demo_scoring_integration():
    """Demonstrate integration with the main scoring pipeline."""
    print("=== DEMO 4: Scoring Pipeline Integration ===")
    
    test_texts = [
        ("Human authentic", "ugh this is taking forever... why is everything so complicated?? anyway i think we should just go with option A and see what happens"),
        ("AI polished", "Based on careful analysis of the available options, I believe option A represents the most viable path forward. This approach offers several distinct advantages while minimizing potential risks.")
    ]
    
    for label, text in test_texts:
        print(f"\n{label}:")
        print(f"Text: '{text[:60]}...'")
        
        # Test old detection method
        old_penalty, old_flag = scoring_detect_llm(text, use_hybrid=False)
        print(f"Heuristic-only: penalty={old_penalty:.3f}, is_llm={old_flag}")
        
        # Test new hybrid method
        new_penalty, new_flag = scoring_detect_llm(text, use_hybrid=True)
        print(f"Hybrid method: penalty={new_penalty:.3f}, is_llm={new_flag}")
        
        # Show difference
        diff = abs(new_penalty - old_penalty)
        print(f"Difference: {diff:.3f}")
    
    print()


def demo_corner_cases():
    """Test corner cases and edge conditions."""
    print("=== DEMO 5: Corner Cases ===")
    
    edge_cases = [
        ("Empty", ""),
        ("Too short", "Hi"),
        ("Numbers only", "123 456 789 000"),
        ("Repetitive", "The the the the the the the the the the."),
        ("No sentences", "word word word word word word word"),
        ("Mixed languages", "Hello world! Bonjour monde! ¡Hola mundo!"),
    ]
    
    for label, text in edge_cases:
        try:
            penalty, is_llm = detect_llm_likeness(text)
            print(f"{label:15} | Penalty: {penalty:.3f} | LLM: {is_llm}")
        except Exception as e:
            print(f"{label:15} | ERROR: {e}")
    
    print()


def demo_performance_comparison():
    """Compare performance of different detection methods."""
    print("=== DEMO 6: Performance Comparison ===")
    
    # Test text with medium complexity
    test_text = """
    The project's implementation strategy involves several key components that work together 
    to achieve the desired outcomes. First, we need to establish a solid foundation through 
    proper planning and analysis. Next, the development phase requires careful attention to 
    detail and adherence to best practices. Finally, thorough testing and validation ensure 
    that the system meets all requirements and performs optimally under various conditions.
    """
    
    import time
    
    # Time heuristic detection
    start_time = time.time()
    heuristic_penalty, heuristic_is_llm = 0.0, False
    for _ in range(100):
        heuristic_penalty, heuristic_is_llm = heuristic_llm_detection(test_text)
    heuristic_time = time.time() - start_time
    
    print(f"Heuristic detection (100 runs):")
    print(f"  Time: {heuristic_time:.4f}s ({heuristic_time*10:.2f}ms per call)")
    print(f"  Result: penalty={heuristic_penalty:.3f}, is_llm={heuristic_is_llm}")
    
    # Time hybrid detection
    start_time = time.time()
    hybrid_penalty, hybrid_is_llm = 0.0, False
    for _ in range(100):
        hybrid_penalty, hybrid_is_llm = detect_llm_likeness(test_text)
    hybrid_time = time.time() - start_time
    
    print(f"\nHybrid detection (100 runs):")
    print(f"  Time: {hybrid_time:.4f}s ({hybrid_time*10:.2f}ms per call)")
    print(f"  Result: penalty={hybrid_penalty:.3f}, is_llm={hybrid_is_llm}")
    
    if hybrid_time > 0:
        speedup = heuristic_time / hybrid_time
        print(f"  Speed ratio: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than heuristic")
    
    print()


if __name__ == "__main__":
    print("Biometric Text Analysis - Hybrid LLM Detection Demo")
    print("=" * 55)
    print()
    
    # Initialize the system
    print("Initializing LLM detection system...")
    initialize_llm_detection()
    print()
    
    # Run all demos
    demo_feature_extraction()
    demo_heuristic_detection()
    demo_hybrid_detection()
    demo_scoring_integration()
    demo_corner_cases() 
    demo_performance_comparison()
    
    print("Demo completed! The hybrid LLM detection system provides:")
    print("✓ 20 advanced linguistic features (sentence variance, POS entropy, perplexity)")
    print("✓ Machine learning classification with GradientBoostingClassifier")
    print("✓ Heuristic fallback for reliability")
    print("✓ Seamless integration with existing scoring pipeline")
    print("✓ Robust handling of edge cases and missing dependencies")
    print("✓ Performance optimization with smart feature caching")