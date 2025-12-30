# tests/test_processor.py
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from candidate_fit_predictor import CVProcessor, JobDescriptionProcessor

class TestProcessors(unittest.TestCase):
    
    def test_cv_name_extraction(self):
        processor = CVProcessor()
        
        # Test various name formats
        test_cases = [
            ("John Doe\nSoftware Engineer", "John Doe"),
            ("NAME: Jane Smith\nSummary:", "Jane Smith"),
            ("Candidate: Bob Johnson\nSkills:", "Bob Johnson"),
            ("This is not a name\nAnother line", "Unknown Candidate"),
        ]
        
        for cv_text, expected_name in test_cases:
            cv_data = processor.parse_cv(cv_text)
            self.assertEqual(cv_data.name, expected_name)
    
    def test_skill_extraction(self):
        processor = CVProcessor()
        
        cv_text = """
        Skills:
        Python, JavaScript, React, AWS, Docker, Machine Learning
        
        Additional Skills:
        SQL, NoSQL, Git, CI/CD
        """
        
        cv_data = processor.parse_cv(cv_text)
        
        # Should find skills in various categories
        self.assertGreater(len(cv_data.skills.get('programming', [])), 0)
        self.assertGreater(len(cv_data.skills.get('cloud_devops', [])), 0)
        self.assertGreater(len(cv_data.skills.get('data_ai', [])), 0)
    
    def test_experience_extraction(self):
        processor = CVProcessor()
        
        test_cases = [
            ("5 years of experience", 5.0),
            ("3+ years experience", 3.0),
            ("Experience: 2 years", 2.0),
            ("2-4 years experience", 4.0),
            ("1 year as developer", 1.0),
        ]
        
        for cv_text, expected_exp in test_cases:
            cv_data = processor.parse_cv(cv_text)
            self.assertEqual(cv_data.experience, expected_exp)
    
    def test_education_extraction(self):
        processor = CVProcessor()
        
        cv_text = """
        Education:
        Bachelor of Science in Computer Science
        Master of Data Science
        PhD in Artificial Intelligence
        """
        
        cv_data = processor.parse_cv(cv_text)
        
        self.assertEqual(len(cv_data.education), 3)
        
        degrees = [edu['degree'] for edu in cv_data.education]
        self.assertIn("Bachelor's", degrees)
        self.assertIn("Master's", degrees)
        self.assertIn("PhD", degrees)
    
    def test_jd_title_extraction(self):
        processor = JobDescriptionProcessor()
        
        test_cases = [
            ("Senior Software Engineer\nRequirements:", "Senior Software Engineer"),
            ("Position: Data Scientist\nDescription:", "Data Scientist"),
            ("Role: DevOps Engineer\nSkills:", "DevOps Engineer"),
            ("Job Title: ML Engineer\nRequired:", "ML Engineer"),
        ]
        
        for jd_text, expected_title in test_cases:
            jd_data = processor.parse_job_description(jd_text)
            self.assertEqual(jd_data.title, expected_title)
    
    def test_jd_skill_extraction(self):
        processor = JobDescriptionProcessor()
        
        jd_text = """
        Requirements:
        - Python programming
        - AWS experience
        - Docker knowledge
        - Machine learning
        
        Required Skills:
        Python, AWS, Docker, TensorFlow
        """
        
        jd_data = processor.parse_job_description(jd_text)
        
        self.assertGreater(len(jd_data.required_skills), 0)
        self.assertIn('python', jd_data.required_skills)
        self.assertIn('aws', jd_data.required_skills)
        self.assertIn('docker', jd_data.required_skills)
    
    def test_jd_experience_requirement(self):
        processor = JobDescriptionProcessor()
        
        test_cases = [
            ("3+ years experience required", 3.0),
            ("2-5 years of experience", 2.0),
            ("Minimum 1 year experience", 1.0),
            ("5 years in software development", 5.0),
        ]
        
        for jd_text, expected_exp in test_cases:
            jd_data = processor.parse_job_description(jd_text)
            self.assertEqual(jd_data.required_experience, expected_exp)
    
    def test_jd_education_requirement(self):
        processor = JobDescriptionProcessor()
        
        test_cases = [
            ("Bachelor's degree required", "Bachelor's"),
            ("Master's degree preferred", "Master's"),
            ("PhD in Computer Science", "PhD"),
            ("No education requirement", "Not specified"),
        ]
        
        for jd_text, expected_edu in test_cases:
            jd_data = processor.parse_job_description(jd_text)
            self.assertEqual(jd_data.required_education, expected_edu)

if __name__ == '__main__':
    unittest.main()