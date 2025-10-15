"""
Demo script showing Step 6 scoring and policy in action.

This demonstrates the complete flow from text input to authentication decision.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scoring import score_sample
from policy import decide, get_default_thresholds


def demo_authentication_flow():
    """Demonstrate the complete scoring and policy flow."""

    print("=" * 70)
    print("BIOMETRIC TEXT AUTHENTICATION - STEP 6 DEMO")
    print("=" * 70)
    print()

    # Simulate a user profile (would normally come from database)
    print("1. Loading user profile...")
    profile = {
        'user_id': 'demo_user_123',
        'centroid': np.random.randn(512).astype(np.float32),
        'cov_diag': np.ones(512, dtype=np.float32) * 0.1,
        'style_mean': np.random.randn(512).astype(np.float32),
        'style_std': np.ones(512, dtype=np.float32) * 0.2,
    }
    # Normalize centroid
    profile['centroid'] = profile['centroid'] / np.linalg.norm(profile['centroid'])
    print(f"   ✓ Profile loaded for user: {profile['user_id']}")
    print()

    # Test scenarios
    scenarios = [
        {
            'name': 'Legitimate User (High Match)',
            'text': "Hey team! Just wanted to follow up on yesterday's meeting. I think we should prioritize the frontend work this sprint, as it's blocking several other features. Let me know your thoughts when you get a chance. Thanks!",
            'embedding_similarity': 0.95,  # Very similar to profile
            'style_similarity': 0.90,
        },
        {
            'name': 'Legitimate User (Medium Match)',
            'text': "Quick update on the project status. We've completed most of the backend implementation and are moving forward with testing. There are a few edge cases we need to address, but overall progress is good. Will send a detailed report by end of day.",
            'embedding_similarity': 0.78,  # Moderate similarity
            'style_similarity': 0.75,
        },
        {
            'name': 'Potential Impostor (Low Match)',
            'text': "I am writing to inform you regarding the current status of the project deliverables. All components have been systematically evaluated and documented according to established protocols. Please review the attached documentation at your earliest convenience.",
            'embedding_similarity': 0.55,  # Low similarity
            'style_similarity': 0.50,
        },
        {
            'name': 'Short Text (Insufficient Length)',
            'text': "Sounds good, thanks!",
            'embedding_similarity': 0.90,
            'style_similarity': 0.85,
        },
    ]

    thresholds = get_default_thresholds()
    print(f"2. Default Thresholds:")
    print(f"   High confidence: {thresholds['high']}")
    print(f"   Medium confidence: {thresholds['med']}")
    print()

    print("3. Testing Authentication Scenarios:")
    print("=" * 70)

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print("-" * 70)

        text = scenario['text']
        word_count = len(text.split())
        print(f"Text: \"{text[:60]}...\"")
        print(f"Word count: {word_count}")

        # Simulate embeddings (in reality, these come from encoder and feature extractor)
        # Create embedding similar to profile based on similarity level
        similarity_target = scenario['embedding_similarity']
        embedding = profile['centroid'] * similarity_target + \
                   np.random.randn(512).astype(np.float32) * (1 - similarity_target) * 0.1
        embedding = embedding / np.linalg.norm(embedding)

        # Create style features
        style_similarity_target = scenario['style_similarity']
        style_features = profile['style_mean'] * style_similarity_target + \
                        np.random.randn(512).astype(np.float32) * (1 - style_similarity_target) * 2.0

        # Score the sample
        result = score_sample(
            profile,
            text,
            embedding,
            style_features
        )

        print(f"\nScoring Results:")
        print(f"  Final Score:     {result['final_score']:.3f}")
        print(f"  Semantic:        {result['semantic_score']:.3f}")
        print(f"  Stylometry:      {result['stylometry_score']:.3f}")
        print(f"  LLM Penalty:     {result['llm_penalty']:.3f}")
        print(f"  LLM-like:        {result['llm_like']}")

        # Make policy decision
        decision = decide(
            score=result['final_score'],
            threshold_high=thresholds['high'],
            threshold_med=thresholds['med'],
            text=text,
            word_count=word_count,
            llm_like=result['llm_like'],
            flow='verify'
        )

        print(f"\nPolicy Decision:")
        print(f"  Decision:        {decision['decision'].upper()}")
        print(f"  Reasons:         {', '.join(decision['reasons'])}")

        # Interpretation
        print(f"\nInterpretation:")
        if decision['decision'] == 'allow':
            print("  ✓ Authentication ALLOWED - User identity confirmed")
        elif decision['decision'] == 'challenge':
            print("  ⚠ Additional CHALLENGE required - Moderate confidence")
        elif decision['decision'] == 'step_up':
            print("  ✗ STEP-UP authentication required - Low confidence")
        else:
            print("  ✗ Authentication DENIED")

        print()

    print("=" * 70)
    print("Demo complete!")
    print()
    print("Summary:")
    print("- High match (score ≥ 0.84) → ALLOW")
    print("- Medium match (0.72 ≤ score < 0.84) → CHALLENGE")
    print("- Low match (score < 0.72) → STEP_UP")
    print("- Short text (< 50 words) → CHALLENGE")
    print("- LLM-like text → CHALLENGE or STEP_UP")
    print("=" * 70)


if __name__ == "__main__":
    demo_authentication_flow()
