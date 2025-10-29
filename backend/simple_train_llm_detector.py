#!/usr/bin/env python3
"""
Simple LLM Detection Model Training Script

This script uses the existing train_llm_detector function from llm_detection.py
to train a model with sample data. It works with the current environment setup.

Usage:
    python simple_train_llm_detector.py
"""

import os
import sys
import numpy as np

# Import our LLM detection module
try:
    from llm_detection import train_llm_detector, SKLEARN_AVAILABLE, load_llm_model, detect_llm_likeness
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn is not available. Please install it with:")
        print("pip install scikit-learn")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import LLM detection module: {e}")
    sys.exit(1)


def generate_enhanced_sample_data(n_samples: int = 500):
    """
    Generate enhanced sample training data for demonstration.
    
    Creates more diverse and realistic examples than the basic version.
    """
    print(f"Generating {n_samples} enhanced sample texts...")
    
    # Enhanced human-written text patterns
    human_texts = []
    human_patterns = [
        "hey what's up? not much happening here lol. just {activity} and {feeling}",
        "So I went to {place} yesterday and saw this crazy thing happen! This {person} was trying to {action}",
        "ugh this {thing} is taking forever... when will it end?? anyway i think we should {decision}",
        "Thanks for {help}! Really appreciate it. Let me know if you need anything else {emotion}",
        "I'm not sure about this {topic} tbh. Maybe we should try {alternative}? What do you think?",
        "The {item} was pretty {quality}, though the {aspect} was kinda {description}. {opinion} though!",
        "Can you believe what happened {when}?! I {reaction} when I saw it",
        "Working {where} has its perks but also challenges. Sometimes it's hard to {challenge}",
        "Just finished {activity} that you recommended. Really {feeling} it, thanks!",
        "The {subject} has been {state} lately. One day it's {condition1}, next day it's {condition2}!",
        "lol that's {reaction}! didn't expect that to happen tbh. {followup}",
        "nah i don't think that's gonna work. maybe try {suggestion} instead? {reasoning}",
        "btw did you hear about {news}? apparently {details} which is kinda {opinion}",
        "sorry for the late reply! been super busy with {activity} and couldn't {reason}",
        "omg {event} was amazing! definitely {recommendation}. you should {suggestion}",
    ]
    
    # Vocabulary for human text generation
    human_vocab = {
        'activity': ['working on stuff', 'watching netflix', 'scrolling social media', 'studying for exams', 'cooking dinner'],
        'feeling': ['pretty tired', 'kinda stressed', 'really excited', 'somewhat confused', 'totally relaxed'],
        'place': ['the store', 'work', 'school', 'the mall', 'downtown'],
        'person': ['guy', 'woman', 'customer', 'student', 'employee'],
        'action': ['return something broken', 'argue with staff', 'cut in line', 'pay with pennies', 'complain loudly'],
        'thing': ['meeting', 'project', 'assignment', 'presentation', 'interview'],
        'decision': ['go with option B', 'try something else', 'wait and see', 'ask for help', 'figure it out'],
        'help': ['your help', 'the advice', 'being there', 'listening', 'your support'],
        'emotion': ['❤️', '😊', '🙏', '👍', ''],
        'topic': ['approach', 'idea', 'plan', 'method', 'strategy'],
        'alternative': ['something different', 'another way', 'plan B', 'a different approach', 'thinking outside the box'],
        'item': ['movie', 'book', 'show', 'game', 'restaurant', 'app'],
        'quality': ['good', 'decent', 'okay', 'nice', 'cool', 'interesting'],
        'aspect': ['ending', 'beginning', 'middle part', 'story', 'characters', 'interface'],
        'description': ['weird', 'off', 'different', 'unexpected', 'confusing', 'predictable'],
        'opinion': ['Worth it', 'Not bad', 'Pretty good', 'Recommended', 'Could be better'],
        'when': ['today', 'yesterday', 'earlier', 'this morning', 'last night'],
        'reaction': ['couldn\'t stop laughing', 'was so surprised', 'couldn\'t believe it', 'was shocked', 'freaked out'],
        'where': ['from home', 'remotely', 'in the office', 'freelance', 'part-time'],
        'challenge': ['focus', 'stay motivated', 'understand everything', 'manage time', 'stay organized'],
        'activity': ['reading that book', 'watching that show', 'trying that recipe', 'using that app', 'playing that game'],
        'subject': ['weather', 'traffic', 'work schedule', 'school', 'internet'],
        'state': ['crazy', 'unpredictable', 'weird', 'stressful', 'chaotic'],
        'condition1': ['super busy', 'really quiet', 'totally hectic', 'pretty chill'],
        'condition2': ['completely different', 'way worse', 'much better', 'exactly the same'],
        'followup': ['totally random', 'so unexpected', 'makes no sense', 'kinda funny actually'],
        'suggestion': ['asking someone else', 'googling it', 'waiting a bit', 'trying later', 'getting help'],
        'reasoning': ['just seems better', 'might work better', 'less risky', 'more practical'],
        'news': ['that thing', 'the update', 'what happened', 'the announcement', 'that situation'],
        'details': ['they changed everything', 'it\'s totally different now', 'nobody saw it coming'],
        'reason': ['get back to you', 'check messages', 'focus on that', 'think about it'],
        'event': ['that concert', 'the game', 'that party', 'the event', 'that show'],
        'recommendation': ['worth going', 'really fun', 'totally worth it', 'amazing experience'],
    }
    
    # Generate human texts
    for _ in range(n_samples // 2):
        pattern = np.random.choice(human_patterns)
        text = pattern
        for key, values in human_vocab.items():
            if f'{{{key}}}' in text:
                text = text.replace(f'{{{key}}}', np.random.choice(values))
        human_texts.append(text)
    
    # Enhanced AI-generated text patterns
    ai_texts = []
    ai_patterns = [
        "Furthermore, it is important to note that {topic} requires careful consideration of {factors}. Additionally, {approach} demonstrates significant advantages over {alternatives}.",
        "Based on comprehensive analysis of {data}, I believe the most effective approach would be to {strategy}. This methodology offers several distinct benefits while minimizing {risks}.",
        "The implementation of {system} demonstrates remarkable capabilities in addressing {challenges}. Moreover, the solution provides comprehensive functionality that meets {requirements}.",
        "In conclusion, the evidence clearly supports the adoption of {solution}. Furthermore, the benefits significantly outweigh {drawbacks}. Therefore, I recommend {action}.",
        "It is essential to consider the various factors that influence {outcomes}. Additionally, proper {processes} must be established to ensure optimal {results}.",
        "The comprehensive evaluation reveals several key insights regarding {domain}. Moreover, the proposed {enhancements} demonstrate significant potential for improving {metrics}.",
        "Based on thorough assessment of {evidence}, it becomes evident that {conclusions}. Furthermore, the implementation of {methods} will ensure {success}.",
        "The strategic approach encompasses multiple dimensions of {scope}. Additionally, the methodology provides a framework for {objectives}. Therefore, {stakeholders} can achieve {goals}.",
        "Comprehensive analysis indicates that the proposed {solution} addresses {needs} effectively. Moreover, the implementation strategy demonstrates alignment with {standards}.",
        "The evaluation process reveals significant opportunities for {improvements}. Furthermore, the systematic approach ensures thorough consideration of {variables}.",
        "It should be noted that {observations} present compelling evidence for {hypotheses}. Consequently, stakeholders can proceed with confidence in {directions}.",
        "The methodology demonstrates exceptional performance across various {applications}. Additionally, the framework provides {capabilities} while maintaining {quality}.",
        "In summary, the proposed solution offers {value} through {mechanisms}. Therefore, implementation of this approach is highly recommended for {purposes}.",
        "The systematic evaluation confirms that {findings} align with {expectations}. Moreover, the results demonstrate {performance} across multiple {dimensions}.",
        "Based on empirical evidence, {option} emerges as the optimal {choice}. Furthermore, this approach ensures {benefits} while addressing {concerns}.",
    ]
    
    # Vocabulary for AI text generation
    ai_vocab = {
        'topic': ['system implementation', 'process optimization', 'strategic planning', 'organizational development'],
        'factors': ['multiple variables', 'diverse parameters', 'key considerations', 'critical elements'],
        'approach': ['this methodology', 'the proposed framework', 'the systematic solution', 'the comprehensive strategy'],
        'alternatives': ['traditional methods', 'existing approaches', 'conventional strategies', 'current practices'],
        'data': ['available information', 'empirical evidence', 'analytical results', 'performance metrics'],
        'strategy': ['implement comprehensive solutions', 'adopt systematic methodologies', 'utilize advanced frameworks'],
        'risks': ['potential complications', 'associated challenges', 'possible limitations', 'operational constraints'],
        'system': ['integrated platform', 'comprehensive framework', 'advanced solution', 'sophisticated architecture'],
        'challenges': ['complex requirements', 'multifaceted problems', 'diverse objectives', 'operational demands'],
        'requirements': ['stakeholder expectations', 'performance standards', 'quality criteria', 'functional specifications'],
        'solution': ['innovative approach', 'strategic framework', 'comprehensive methodology', 'systematic implementation'],
        'drawbacks': ['potential limitations', 'associated risks', 'possible constraints', 'implementation challenges'],
        'action': ['proceeding with deployment', 'adopting this framework', 'implementing the strategy'],
        'outcomes': ['system performance', 'operational efficiency', 'strategic objectives', 'organizational goals'],
        'processes': ['monitoring procedures', 'evaluation protocols', 'quality assurance measures'],
        'results': ['optimal performance', 'strategic success', 'operational excellence', 'desired outcomes'],
        'domain': ['operational sphere', 'strategic landscape', 'organizational context', 'business environment'],
        'enhancements': ['optimization strategies', 'improvement methodologies', 'advancement techniques'],
        'metrics': ['performance indicators', 'success measures', 'efficiency ratings', 'quality benchmarks'],
        'evidence': ['analytical data', 'empirical findings', 'research results', 'performance assessments'],
        'conclusions': ['strategic insights', 'operational improvements', 'systematic enhancements'],
        'methods': ['best practices', 'proven methodologies', 'established protocols', 'standard procedures'],
        'success': ['optimal outcomes', 'strategic achievements', 'operational excellence', 'performance targets'],
        'scope': ['organizational development', 'strategic enhancement', 'operational improvement'],
        'objectives': ['continuous improvement', 'strategic advancement', 'performance optimization'],
        'stakeholders': ['organizations', 'management teams', 'project leaders', 'decision makers'],
        'goals': ['strategic objectives', 'operational targets', 'performance milestones', 'success criteria'],
        'needs': ['operational requirements', 'strategic demands', 'performance expectations'],
        'standards': ['industry best practices', 'quality benchmarks', 'operational protocols'],
        'improvements': ['systematic optimization', 'strategic enhancement', 'performance advancement'],
        'variables': ['operational factors', 'strategic elements', 'performance parameters'],
        'observations': ['empirical findings', 'analytical insights', 'systematic evaluations'],
        'hypotheses': ['strategic assumptions', 'operational theories', 'performance projections'],
        'directions': ['strategic approaches', 'operational methodologies', 'systematic implementations'],
        'applications': ['operational contexts', 'strategic environments', 'implementation scenarios'],
        'capabilities': ['advanced functionality', 'enhanced performance', 'superior efficiency'],
        'quality': ['operational standards', 'performance benchmarks', 'excellence criteria'],
        'value': ['strategic benefits', 'operational advantages', 'performance improvements'],
        'mechanisms': ['systematic processes', 'structured methodologies', 'coordinated approaches'],
        'purposes': ['achieving objectives', 'meeting requirements', 'attaining goals'],
        'findings': ['analytical results', 'empirical outcomes', 'evaluation conclusions'],
        'expectations': ['projected results', 'anticipated outcomes', 'strategic targets'],
        'performance': ['operational efficiency', 'strategic effectiveness', 'systematic success'],
        'dimensions': ['operational areas', 'strategic domains', 'performance sectors'],
        'option': ['strategic alternative', 'systematic approach', 'comprehensive solution'],
        'choice': ['preferred strategy', 'optimal methodology', 'recommended approach'],
        'benefits': ['strategic advantages', 'operational improvements', 'performance enhancements'],
        'concerns': ['potential challenges', 'operational considerations', 'strategic limitations'],
    }
    
    # Generate AI texts
    for _ in range(n_samples // 2):
        pattern = np.random.choice(ai_patterns)
        text = pattern
        for key, values in ai_vocab.items():
            if f'{{{key}}}' in text:
                text = text.replace(f'{{{key}}}', np.random.choice(values))
        ai_texts.append(text)
    
    # Combine texts and labels
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    
    # Shuffle
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    print(f"Generated {len(human_texts)} human and {len(ai_texts)} AI samples")
    return list(texts), list(labels)


def main():
    """Main training function."""
    print("=" * 60)
    print("Simple LLM Detection Model Training Script")
    print("=" * 60)
    
    # Generate enhanced sample data
    print("\nGenerating enhanced training data...")
    texts, labels = generate_enhanced_sample_data(n_samples=1000)
    print(f"Created {len(texts)} samples ({sum(1 for l in labels if l == 0)} human, {sum(1 for l in labels if l == 1)} AI)")
    
    # Create models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created {models_dir} directory")
    
    # Train the model
    print("\nTraining the LLM detection model...")
    try:
        metrics = train_llm_detector(
            texts=texts,
            labels=labels,
            model_path="models/llm_detector.pkl",
            test_size=0.2,  # Use 20% for testing
            random_state=42
        )
        
        print("\nTraining Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC:       {metrics['auc']:.3f}")
        print(f"  Samples:   {metrics['n_samples']}")
        print(f"  Features:  {metrics['n_features']}")
        
        print(f"\nModel saved to: models/llm_detector.pkl")
        print("You can now use the hybrid detection system with ML classification!")
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return False
    
    # Test the trained model
    print("\nTesting the trained model...")
    try:
        # Load the trained model
        load_llm_model("models/llm_detector.pkl")
        
        # Test samples
        test_samples = [
            ("Human casual", "hey that's pretty cool! didn't know that was possible lol. gonna try it later for sure"),
            ("Human story", "so yesterday i went to the store and this crazy thing happened. this guy was trying to return a watermelon he'd already eaten! the cashier was so confused"),
            ("AI formal", "Furthermore, it is essential to consider the comprehensive implications of this systematic approach. Additionally, the implementation demonstrates significant advantages over existing methodologies while ensuring optimal performance."),
            ("AI trying casual", "I think this approach is really interesting and beneficial. Additionally, it offers many comprehensive advantages. However, there are some systematic challenges to consider as well."),
        ]
        
        print("Test Results:")
        for label, text in test_samples:
            penalty, is_llm = detect_llm_likeness(text, use_ml=True)
            status = "🤖 AI-like" if is_llm else "👤 Human-like"
            print(f"  {label:15} | penalty={penalty:.3f} | {status}")
        
        print("\nTraining completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Testing failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS! Next steps:")
        print("1. The model is now saved and ready to use")
        print("2. The hybrid detection system will automatically use it")
        print("3. Collect more diverse training data for better accuracy")
        print("4. Consider tuning hyperparameters for optimal performance")
        print("5. Integrate with your biometric authentication system")
        print("=" * 60)
    else:
        print("\nTraining failed. Please check the error messages above.")
        sys.exit(1)