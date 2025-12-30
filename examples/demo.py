# examples/demo.py
"""
Demo script for Candidate Fit Predictor.
Shows various use cases and features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from candidate_fit_predictor import CandidateFitPredictor

def demo_basic():
    """Basic prediction demo."""
    print("="*60)
    print("DEMO 1: Basic Prediction")
    print("="*60)
    
    cv_text = """
    Sarah Johnson
    Data Scientist
    
    Summary:
    Data Scientist with 4 years experience in ML and analytics.
    
    Skills:
    Python, Machine Learning, SQL, TensorFlow, AWS, Statistics
    
    Experience:
    4 years as Data Scientist at Analytics Corp
    
    Education:
    Master of Science in Data Science
    """
    
    jd_text = """
    Senior Data Scientist
    
    Requirements:
    - 3+ years experience in data science
    - Strong Python and ML skills
    - Experience with TensorFlow
    - Master's degree preferred
    
    Required Skills:
    Python, Machine Learning, TensorFlow, SQL, Statistics
    """
    
    predictor = CandidateFitPredictor()
    score, analysis = predictor.predict(cv_text, jd_text)
    
    print(f"\nScore: {score}%")
    print(f"Interpretation: {analysis['interpretation']}")
    print(f"Skill Match: {analysis['skill_analysis']['match_percentage']}%")
    
    return score, analysis

def demo_multiple_candidates():
    """Demo screening multiple candidates."""
    print("\n" + "="*60)
    print("DEMO 2: Screening Multiple Candidates")
    print("="*60)
    
    jd_text = """
    Software Engineer
    
    Requirements:
    - 2+ years Python experience
    - Django framework
    - React.js
    - AWS basics
    
    Required Skills:
    Python, Django, React, AWS
    """
    
    candidates = [
        {
            'name': 'Alice',
            'cv': """
            Alice Chen
            Software Engineer
            
            Skills:
            Python, Django, React, AWS, Docker
            
            Experience:
            3 years at Tech Company
            
            Education:
            Bachelor in Computer Science
            """
        },
        {
            'name': 'Bob',
            'cv': """
            Bob Smith
            Developer
            
            Skills:
            Java, Spring, MySQL
            
            Experience:
            4 years at Enterprise Inc
            
            Education:
            Bachelor in IT
            """
        },
        {
            'name': 'Charlie',
            'cv': """
            Charlie Brown
            Full Stack Developer
            
            Skills:
            Python, JavaScript, HTML, CSS
            
            Experience:
            1 year internship
            
            Education:
            Computer Science Student
            """
        }
    ]
    
    predictor = CandidateFitPredictor()
    
    results = []
    for candidate in candidates:
        score, analysis = predictor.predict(candidate['cv'], jd_text)
        results.append({
            'name': candidate['name'],
            'score': score,
            'match': analysis['skill_analysis']['match_percentage']
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nCandidate Rankings:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']}: {result['score']:.1f}% "
              f"(Skill match: {result['match']:.1f}%)")
    
    return results

def demo_cv_optimization():
    """Demo CV optimization suggestions."""
    print("\n" + "="*60)
    print("DEMO 3: CV Optimization Recommendations")
    print("="*60)
    
    cv_text = """
    David Wilson
    Software Developer
    
    Skills:
    Python, Java, C++
    
    Experience:
    2 years software development
    
    Education:
    Bachelor in Computer Science
    """
    
    jd_text = """
    Python Backend Developer
    
    Requirements:
    - Python expertise
    - Django/Flask experience
    - REST API development
    - Docker and AWS knowledge
    - 2+ years experience
    
    Required Skills:
    Python, Django, Flask, REST APIs, Docker, AWS
    """
    
    predictor = CandidateFitPredictor()
    score, analysis = predictor.predict(cv_text, jd_text)
    
    print(f"\nCurrent Score: {score}%")
    print("\nMissing Skills:")
    for i, skill in enumerate(analysis['skill_analysis']['missing_skills'], 1):
        print(f"  {i}. {skill}")
    
    print("\nRecommendations to improve score:")
    for i, rec in enumerate(analysis['detailed_recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return analysis

def demo_batch_processing():
    """Demo batch processing with CSV."""
    print("\n" + "="*60)
    print("DEMO 4: Batch Processing Simulation")
    print("="*60)
    
    import pandas as pd
    from io import StringIO
    
    # Simulate CSV data
    csv_data = """id,name,cv_text,jd_text
    1,Emma,"Skills: Python, ML","Requirements: Python, ML"
    2,Liam,"Skills: Java, Spring","Requirements: Python, ML"
    3,Olivia,"Skills: Python, Django, AWS","Requirements: Python, ML"
    """
    
    df = pd.read_csv(StringIO(csv_data))
    predictor = CandidateFitPredictor()
    
    predictions = []
    for _, row in df.iterrows():
        score, analysis = predictor.predict(row['cv_text'], row['jd_text'])
        predictions.append({
            'id': row['id'],
            'name': row['name'],
            'score': score,
            'recommendation': 'Interview' if score >= 70 else 'Review'
        })
    
    results_df = pd.DataFrame(predictions)
    
    print("\nBatch Results:")
    print(results_df.to_string(index=False))
    
    return results_df

def demo_model_training():
    """Demo model training and evaluation."""
    print("\n" + "="*60)
    print("DEMO 5: Model Training")
    print("="*60)
    
    predictor = CandidateFitPredictor()
    
    print("Training model with 100 samples...")
    metrics = predictor.train(n_samples=100)
    
    print("\nPerformance Metrics:")
    print(f"MAE: {metrics['mae']:.2f}%")
    print(f"R²: {metrics['r2']:.3f}")
    print(f"CV R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    
    # Save the model
    predictor.save_model("demo_model.pkl")
    print("\nModel saved to 'demo_model.pkl'")
    
    return metrics

def main():
    """Run all demos."""
    print("Candidate Fit Predictor - Demo Suite")
    print("="*60)
    
    # Run demos
    demo_basic()
    demo_multiple_candidates()
    demo_cv_optimization()
    demo_batch_processing()
    demo_model_training()
    
    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()