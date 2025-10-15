"""Quick integration test for Step 6 components."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scoring import cosine_similarity, score_sample
from policy import decide, get_default_thresholds

print('Testing Step 6 Components:')
print('=' * 60)

# Test 1: Cosine similarity
print('\n1. Cosine Similarity Test')
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([1.0, 0.0, 0.0])
sim = cosine_similarity(v1, v2)
print(f'   Identical vectors: {sim:.3f} (expected: 1.000)')
assert abs(sim - 1.0) < 0.001, 'Cosine similarity failed'
print('   ✓ PASSED')

# Test 2: Policy decision
print('\n2. Policy Decision Test')
thresholds = get_default_thresholds()
print(f'   Thresholds: high={thresholds["high"]}, med={thresholds["med"]}')

result = decide(0.90, 0.84, 0.72, 'test ' * 60, 60)
print(f'   High score (0.90) → {result["decision"]} (expected: allow)')
assert result['decision'] == 'allow', 'High confidence decision failed'
print('   ✓ PASSED')

result = decide(0.78, 0.84, 0.72, 'test ' * 60, 60)
print(f'   Med score (0.78) → {result["decision"]} (expected: challenge)')
assert result['decision'] == 'challenge', 'Medium confidence decision failed'
print('   ✓ PASSED')

result = decide(0.65, 0.84, 0.72, 'test ' * 60, 60)
print(f'   Low score (0.65) → {result["decision"]} (expected: step_up)')
assert result['decision'] == 'step_up', 'Low confidence decision failed'
print('   ✓ PASSED')

# Test 3: Score sample integration
print('\n3. Score Sample Integration Test')
profile = {
    'centroid': np.random.randn(512).astype(np.float32),
    'style_mean': np.random.randn(512).astype(np.float32),
    'style_std': np.ones(512, dtype=np.float32) * 0.1,
}
profile['centroid'] /= np.linalg.norm(profile['centroid'])

embedding = profile['centroid'].copy()
style = profile['style_mean'].copy()
text = 'Test message ' * 20

result = score_sample(profile, text, embedding, style)
print(f'   Final score: {result["final_score"]:.3f}')
print(f'   Semantic: {result["semantic_score"]:.3f}')
print(f'   Stylometry: {result["stylometry_score"]:.3f}')
assert 'final_score' in result, 'Score sample failed'
assert 0.0 <= result['final_score'] <= 1.0, 'Score out of range'
print('   ✓ PASSED')

print('\n' + '=' * 60)
print('All Step 6 components working correctly! ✓')
