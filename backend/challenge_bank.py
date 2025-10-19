"""
Challenge bank for enrollment and verification flows.

Provides randomized writing prompts with constraints to collect
natural writing samples from users.
"""
import random
from typing import List, Dict
from schemas import ChallengeInfo


# Challenge prompts organized by category
CHALLENGE_PROMPTS = {
    "work": [
        {
            "id": "work_update_1",
            "prompt": "Write a brief update to your team about a project you're working on. Include current progress and next steps.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "work_meeting_1",
            "prompt": "Summarize the key points from a recent meeting or discussion you had. What were the main takeaways?",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "work_email_1",
            "prompt": "Write an email to a colleague about scheduling a meeting or asking for help with something.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "work_feedback_1",
            "prompt": "Provide feedback or thoughts on a proposal, document, or idea that was shared with you recently.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
    ],
    "personal": [
        {
            "id": "personal_day_1",
            "prompt": "Describe what you did today or plan to do tomorrow. Include some specific details.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "personal_hobby_1",
            "prompt": "Tell someone about a hobby, interest, or activity you enjoy. What makes it interesting to you?",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "personal_weekend_1",
            "prompt": "Write about your plans for the weekend or describe a recent weekend activity you enjoyed.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "personal_recommendation_1",
            "prompt": "Recommend a book, movie, restaurant, or place you've enjoyed recently. Explain why you liked it.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
    ],
    "opinion": [
        {
            "id": "opinion_topic_1",
            "prompt": "Share your thoughts on a current event, news story, or topic you've been following.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "opinion_decision_1",
            "prompt": "Describe a decision you made recently and explain your reasoning behind it.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "opinion_preference_1",
            "prompt": "Explain why you prefer something over an alternative (e.g., tool, method, approach, product).",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
    ],
    "descriptive": [
        {
            "id": "descriptive_place_1",
            "prompt": "Describe your workspace, home office, or a place where you spend a lot of time.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "descriptive_process_1",
            "prompt": "Explain how you approach a typical task or routine in your daily work or life.",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
        {
            "id": "descriptive_experience_1",
            "prompt": "Describe a recent experience, event, or situation you encountered. What happened?",
            "min_words": 70,
            "timebox_s": 180,
            "constraints": ["no_paste", "min_sentences_3"]
        },
    ],
}


def get_all_challenges() -> List[Dict]:
    """
    Get all available challenges as a flat list.

    Returns:
        List of challenge dictionaries
    """
    all_challenges = []
    for category, challenges in CHALLENGE_PROMPTS.items():
        for challenge in challenges:
            challenge_copy = challenge.copy()
            challenge_copy["category"] = category
            all_challenges.append(challenge_copy)
    return all_challenges


def select_challenges(
    num_challenges: int = 8,
    categories: List[str] = None,
    seed: int = None
) -> List[ChallengeInfo]:
    """
    Select a randomized set of challenges for enrollment.

    Args:
        num_challenges: Number of challenges to select (default: 8)
        categories: List of categories to select from (default: all)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of ChallengeInfo objects
    """
    if seed is not None:
        random.seed(seed)

    # Get challenges from specified categories or all
    if categories:
        available = []
        for cat in categories:
            if cat in CHALLENGE_PROMPTS:
                available.extend(CHALLENGE_PROMPTS[cat])
    else:
        available = get_all_challenges()

    # Ensure we have enough challenges
    if len(available) < num_challenges:
        # Repeat challenges if needed
        available = available * ((num_challenges // len(available)) + 1)

    # Randomly select without replacement
    selected = random.sample(available, num_challenges)

    # Convert to ChallengeInfo objects
    challenge_infos = []
    for challenge in selected:
        challenge_infos.append(
            ChallengeInfo(
                challenge_id=challenge["id"],
                prompt=challenge["prompt"],
                min_words=challenge["min_words"],
                timebox_s=challenge["timebox_s"],
                constraints=challenge["constraints"]
            )
        )

    return challenge_infos


def get_challenge_by_id(challenge_id: str) -> Dict:
    """
    Get a specific challenge by ID.

    Args:
        challenge_id: Challenge identifier

    Returns:
        Challenge dictionary or None if not found
    """
    all_challenges = get_all_challenges()
    for challenge in all_challenges:
        if challenge["id"] == challenge_id:
            return challenge
    return None


def validate_challenge_response(
    challenge_id: str,
    text: str,
    min_words_override: int = None
) -> tuple[bool, List[str]]:
    """
    Validate a challenge response meets requirements.

    Args:
        challenge_id: Challenge identifier
        text: User's response text
        min_words_override: Override minimum words (optional)

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    challenge = get_challenge_by_id(challenge_id)
    if not challenge:
        issues.append("invalid_challenge_id")
        return False, issues

    # Check word count
    word_count = len(text.split())
    min_words = min_words_override or challenge["min_words"]

    if word_count < min_words:
        issues.append(f"insufficient_words_{word_count}_of_{min_words}")

    # Check minimum sentences if required
    if "min_sentences_3" in challenge.get("constraints", []):
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if len(sentences) < 3:
            issues.append("insufficient_sentences")

    is_valid = len(issues) == 0
    return is_valid, issues


def select_mfa_challenge(enrolled_challenge_ids: List[str], seed: int = None) -> ChallengeInfo:
    """
    Select a challenge for MFA verification that is:
    1. From the same category as one of the enrolled challenges (for semantic relevance)
    2. NOT one of the enrolled challenges (to prevent simple memorization)
    
    Args:
        enrolled_challenge_ids: List of challenge IDs used during enrollment
        seed: Random seed for reproducibility (optional)
    
    Returns:
        ChallengeInfo for MFA verification
    """
    if seed is not None:
        random.seed(seed)
    
    # Get categories used in enrollment
    enrolled_categories = set()
    for challenge_id in enrolled_challenge_ids:
        challenge = get_challenge_by_id(challenge_id)
        if challenge:
            enrolled_categories.add(challenge.get("category", ""))
    
    # Get all challenges from those categories
    candidate_challenges = []
    for category in enrolled_categories:
        if category in CHALLENGE_PROMPTS:
            for challenge in CHALLENGE_PROMPTS[category]:
                # Exclude challenges that were used in enrollment
                if challenge["id"] not in enrolled_challenge_ids:
                    candidate_challenges.append(challenge)
    
    # If no candidates (user enrolled with all prompts in categories), fall back to any unused challenge
    if not candidate_challenges:
        all_challenges = get_all_challenges()
        candidate_challenges = [c for c in all_challenges if c["id"] not in enrolled_challenge_ids]
    
    # If still no candidates (edge case), just pick a random one from enrollment
    if not candidate_challenges:
        # This should rarely happen - just reuse a random enrolled challenge
        fallback_challenge = get_challenge_by_id(random.choice(enrolled_challenge_ids))
        return ChallengeInfo(
            challenge_id=fallback_challenge["id"],
            prompt=fallback_challenge["prompt"],
            min_words=fallback_challenge["min_words"],
            timebox_s=fallback_challenge["timebox_s"],
            constraints=fallback_challenge["constraints"]
        )
    
    # Select a random candidate
    selected = random.choice(candidate_challenges)
    
    return ChallengeInfo(
        challenge_id=selected["id"],
        prompt=selected["prompt"],
        min_words=selected["min_words"],
        timebox_s=selected["timebox_s"],
        constraints=selected["constraints"]
    )

