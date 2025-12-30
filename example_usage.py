# example_usage.py

from candidate_fit_predictor import CandidateFitPredictor, create_sample_data
import json

def example_usage():
    """Example usage of the Candidate Fit Predictor."""
    
    # Option 1: Train a new model
    print("Option 1: Training a new model")
    print("-" * 40)
    
    # Create sample data
    candidates, jobs, scores = create_sample_data(100)
    
    # Initialize predictor
    predictor = CandidateFitPredictor(use_advanced_features=True)
    
    # Prepare and train
    X, y = predictor.prepare_training_data(candidates, jobs, scores)
    predictor.train(X, y, model_type='random_forest')
    
    # Save model
    predictor.save_model('trained_model.joblib')
    
    # Option 2: Load existing model and predict
    print("\nOption 2: Loading model and making predictions")
    print("-" * 40)
    
    # Load the trained model
    predictor2 = CandidateFitPredictor()
    predictor2.load_model('trained_model.joblib')
    
    # Example candidate and job
    candidate_example = {
        'id': 'CAND_001',
        'skills': 'Python, Machine Learning, TensorFlow, SQL, AWS',
        'experience': '5 years as a Data Scientist at Tech Company',
        'education': 'Master in Computer Science',
        'location': 'San Francisco',
        'expected_salary': 120000
    }
    
    job_example = {
        'id': 'JOB_001',
        'title': 'Senior Data Scientist',
        'required_skills': 'Python, Machine Learning, Deep Learning, SQL, Statistics',
        'responsibilities': 'Develop ML models, analyze data, deploy solutions',
        'requirements': '3+ years experience, Master degree required',
        'location': 'San Francisco',
        'salary_range': '$100,000 - $140,000'
    }
    
    # Make prediction
    result = predictor2.predict(candidate_example, job_example, return_explanation=True)
    
    print(f"\nPrediction Result:")
    print(f"  Candidate: {result['candidate_id']}")
    print(f"  Job: {result['job_id']}")
    print(f"  Fit Score: {result['fit_score']:.1f}%")
    
    if 'explanation' in result:
        print("\n  Top Matching Factors:")
        for exp in result['explanation']:
            print(f"    {exp['feature']}: {exp['value']:.3f}")
    
    # Option 3: Batch predictions
    print("\nOption 3: Batch predictions")
    print("-" * 40)
    
    # Create multiple examples
    candidates_batch = [
        {
            'id': 'CAND_002',
            'skills': 'Java, Spring Boot, Microservices, Docker',
            'experience': '3 years Backend Developer',
            'education': 'Bachelor in Software Engineering'
        },
        {
            'id': 'CAND_003',
            'skills': 'React, JavaScript, CSS, HTML, Node.js',
            'experience': '2 years Frontend Developer',
            'education': 'Bachelor in Computer Science'
        }
    ]
    
    jobs_batch = [
        {
            'id': 'JOB_002',
            'title': 'Backend Developer',
            'required_skills': 'Java, Spring Boot, REST APIs, Microservices',
            'requirements': '2+ years Java experience'
        },
        {
            'id': 'JOB_003',
            'title': 'Frontend Developer',
            'required_skills': 'React, JavaScript, CSS, HTML5',
            'requirements': '1+ years React experience'
        }
    ]
    
    print("\nBatch Predictions:")
    for candidate in candidates_batch:
        for job in jobs_batch:
            result = predictor2.predict(candidate, job, return_explanation=False)
            print(f"  {candidate['id']} -> {job['id']}: {result['fit_score']:.1f}%")
    
    # Option 4: Save results to JSON
    print("\nOption 4: Exporting results")
    print("-" * 40)
    
    all_results = []
    for candidate in candidates_batch:
        for job in jobs_batch:
            result = predictor2.predict(candidate, job, return_explanation=True)
            all_results.append(result)
    
    # Save to file
    with open('predictions.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Results saved to predictions.json")
    
    return predictor2

if __name__ == "__main__":
    predictor = example_usage()