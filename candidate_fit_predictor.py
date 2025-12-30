"""
Candidate Fit Score Predictor
=============================

A machine learning system that analyzes candidate CVs/resumes against job descriptions
to predict compatibility scores (0-100%) with detailed explanations.

Author: AI Assistant
Date: 2024
Version: 2.0

Features:
- Text similarity analysis using TF-IDF and cosine similarity
- Skill matching with comprehensive skill database
- Experience and education level matching
- ML model (Random Forest) for accurate predictions
- Detailed analysis with recommendations
- No external file dependencies (works with text input)

Requirements:
i am giving requirement separately

Usage:
    python candidate_fit_predictor.py
"""

import pandas as pd
import numpy as np
import re
import json
import warnings
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Basic data processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class SimpleCVProcessor:
    """
    CV Processor for extracting information from CV/resume text.

    This class processes CV text to extract:
    - Skills (technical and soft skills)
    - Years of experience
    - Education details
    - Basic personal information

    Attributes:
        lemmatizer: WordNetLemmatizer for text normalization
        stop_words: Set of English stopwords
        skill_database: Dictionary of categorized skills
    """

    def __init__(self):
        """Initialize CV processor with skill database and NLP tools."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Enhanced skill database categorized by domain
        self.skill_database = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
                          'swift', 'kotlin', 'go', 'rust', 'typescript', 'scala', 'r',
                          'matlab', 'perl', 'bash', 'shell'],
            'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django',
                       'flask', 'spring', 'express', 'laravel', 'asp.net',
                       'ruby on rails', 'jquery', 'bootstrap', 'sass', 'less'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
                         'sqlite', 'cassandra', 'dynamodb', 'elasticsearch',
                         'firebase', 'mariadb'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                     'ansible', 'jenkins', 'git', 'ci/cd', 'devops', 'linux', 'unix'],
            'data_science': ['machine learning', 'deep learning', 'artificial intelligence',
                           'nlp', 'computer vision', 'tensorflow', 'pytorch',
                           'scikit-learn', 'keras', 'pandas', 'numpy', 'matplotlib',
                           'seaborn', 'spark', 'hadoop', 'tableau', 'power bi',
                           'excel', 'statistics'],
            'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving',
                           'critical thinking', 'adaptability', 'time management',
                           'creativity', 'collaboration', 'project management']
        }

    def parse_cv_text(self, cv_text: str) -> Dict[str, Any]:
        """
        Parse CV text and extract structured information.

        Args:
            cv_text: Raw CV text content

        Returns:
            Dictionary containing:
                - raw_text: Original text (lowercased)
                - name: Extracted name (from first line)
                - skills: Dictionary of categorized skills
                - experience: Total years of experience
                - education: List of education degrees

        Example:
            >>> processor = SimpleCVProcessor()
            >>> cv_data = processor.parse_cv_text("John Doe\\nSkills: Python, SQL...")
            >>> print(cv_data['name'])  # "John Doe"
        """
        cv_text = cv_text.lower()

        # Extract various components
        skills = self._extract_skills(cv_text)
        experience = self._extract_experience(cv_text)
        education = self._extract_education(cv_text)

        # Extract name from first line (simple heuristic)
        lines = cv_text.strip().split('\n')
        name = lines[0].title() if lines and len(lines[0].split()) <= 4 else "Unknown"

        return {
            'raw_text': cv_text,
            'name': name,
            'skills': skills,
            'experience': experience,
            'education': education
        }

    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text using skill database.

        Args:
            text: Input text to search for skills

        Returns:
            Dictionary mapping skill categories to found skills
        """
        found_skills = {category: [] for category in self.skill_database.keys()}

        for category, skill_list in self.skill_database.items():
            for skill in skill_list:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    found_skills[category].append(skill)

        return found_skills

    def _extract_experience(self, text: str) -> float:
        """
        Extract total years of experience using regex patterns.

        Args:
            text: Text containing experience information

        Returns:
            Maximum years of experience found (float)
        """
        patterns = [
            r'(\d+)\s*(?:year|yr|years|yrs)',
            r'experience\s*(?:of)?\s*(\d+)',
            r'(\d+)\+?\s*years?\s*experience',
            r'(\d+)\s*-\s*(\d+)\s*years?'
        ]

        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        for m in match:
                            if m.isdigit():
                                max_years = max(max_years, int(m))
                    elif match.isdigit():
                        max_years = max(max_years, int(match))

        return float(max_years)

    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """
        Extract education degrees from text.

        Args:
            text: Text containing education information

        Returns:
            List of dictionaries with degree and field
        """
        education = []

        degree_patterns = [
            r'(bachelor|b\.?s\.?c?|b\.?a\.?|b\.?tech)\s*(?:in|of)?\s*([A-Za-z\s&]+)',
            r'(master|m\.?s\.?c?|m\.?a\.?|m\.?tech|mba)\s*(?:in|of)?\s*([A-Za-z\s&]+)',
            r'(ph\.?d\.?|doctorate)\s*(?:in|of)?\s*([A-Za-z\s&]+)'
        ]

        for pattern in degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                education.append({
                    'degree': match.group(1).title(),
                    'field': match.group(2).title() if match.group(2) else 'Not specified'
                })

        return education


class SimpleJobProcessor:
    """
    Job Description Processor for extracting requirements from job postings.

    This class processes job description text to extract:
    - Required skills
    - Required years of experience
    - Required education level
    - Job title

    Attributes:
        skill_database: Reference to CV processor's skill database
    """

    def __init__(self):
        """Initialize job processor with skill database."""
        self.skill_database = SimpleCVProcessor().skill_database

    def parse_job_description(self, jd_text: str) -> Dict[str, Any]:
        """
        Parse job description text.

        Args:
            jd_text: Raw job description text

        Returns:
            Dictionary containing:
                - raw_text: Original text (lowercased)
                - title: Job title (from first line)
                - skills: List of required skills
                - experience_required: Required years of experience
                - education_required: Required education level
        """
        jd_text = jd_text.lower()

        skills = self._extract_skills(jd_text)
        exp_required = self._extract_experience_requirement(jd_text)
        edu_required = self._extract_education_requirement(jd_text)

        lines = jd_text.strip().split('\n')
        title = lines[0].title() if lines else "Unknown Position"

        return {
            'raw_text': jd_text,
            'title': title,
            'skills': skills,
            'experience_required': exp_required,
            'education_required': edu_required
        }

    def _extract_skills(self, text: str) -> List[str]:
        """
        Extract required skills from job description.

        Args:
            text: Job description text

        Returns:
            List of unique skills found
        """
        skills = []
        cv_processor = SimpleCVProcessor()
        skill_dict = cv_processor._extract_skills(text)

        for category, skill_list in skill_dict.items():
            skills.extend(skill_list)

        return list(set(skills))

    def _extract_experience_requirement(self, text: str) -> float:
        """
        Extract required years of experience.

        Args:
            text: Job description text

        Returns:
            Required years of experience (float)
        """
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
            r'(\d+)\s*-\s*(\d+)\s*years?\s*experience',
            r'minimum\s*(\d+)\s*years?',
            r'(\d+)\s*years?\s*in\s*.*\s*experience'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    return float(matches[0][0])
                else:
                    return float(matches[0])

        return 0.0

    def _extract_education_requirement(self, text: str) -> str:
        """
        Extract required education level.

        Args:
            text: Job description text

        Returns:
            Required education level as string
        """
        patterns = [
            r'(bachelor|b\.?s\.?|b\.?a\.?)\s*(?:degree)?',
            r'(master|m\.?s\.?|m\.?a\.?|mba)\s*(?:degree)?',
            r'(ph\.?d\.?|doctorate)\s*(?:degree)?'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).title()

        return "Not specified"


class CandidateFitPredictor:
    """
    Main ML model for predicting candidate-job compatibility.

    This class implements:
    - Feature extraction from CV and job data
    - ML model training (Random Forest)
    - Prediction with detailed analysis
    - Performance evaluation

    Attributes:
        cv_processor: Instance of SimpleCVProcessor
        jd_processor: Instance of SimpleJobProcessor
        tfidf_vectorizer: TF-IDF vectorizer for text similarity
        feature_scaler: StandardScaler for feature normalization
        model: Trained RandomForestRegressor
        is_trained: Boolean indicating if model is trained
    """

    def __init__(self):
        """Initialize the predictor with processors and ML components."""
        self.cv_processor = SimpleCVProcessor()
        self.jd_processor = SimpleJobProcessor()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500)
        self.feature_scaler = StandardScaler()
        self.model = None
        self.is_trained = False

    def extract_features(self, cv_data: Dict, jd_data: Dict) -> Dict[str, float]:
        """
        Extract numerical features from CV and job data.

        Args:
            cv_data: Parsed CV data
            jd_data: Parsed job description data

        Returns:
            Dictionary of feature names and values

        Features extracted:
            - text_similarity: Cosine similarity between CV and JD texts
            - skill_match_ratio: Percentage of required skills matched
            - cv_experience: Candidate's years of experience
            - experience_gap: Difference between candidate and required experience
            - education_match: Education level comparison (0-1)
            - And 10+ other relevant features
        """
        features = {}

        # 1. Text similarity using TF-IDF and cosine similarity
        cv_text = cv_data['raw_text']
        jd_text = jd_data['raw_text']

        if cv_text and jd_text:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([cv_text, jd_text])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                features['text_similarity'] = cosine_sim
            except:
                features['text_similarity'] = 0

        # 2. Skill matching features
        cv_skills_flat = []
        for category, skills in cv_data['skills'].items():
            cv_skills_flat.extend(skills)

        jd_skills = jd_data['skills']

        if jd_skills:
            matched_skills = set(cv_skills_flat) & set(jd_skills)
            features['skill_match_ratio'] = len(matched_skills) / len(jd_skills)
            features['matched_skills_count'] = len(matched_skills)
            features['total_jd_skills'] = len(jd_skills)
        else:
            features['skill_match_ratio'] = 0
            features['matched_skills_count'] = 0
            features['total_jd_skills'] = 0

        features['total_cv_skills'] = len(cv_skills_flat)

        # 3. Experience matching features
        cv_exp = cv_data['experience']
        jd_exp_req = jd_data['experience_required']

        features['cv_experience'] = cv_exp
        features['jd_required_experience'] = jd_exp_req
        features['experience_gap'] = cv_exp - jd_exp_req
        features['experience_sufficiency'] = 1 if cv_exp >= jd_exp_req else cv_exp / jd_exp_req if jd_exp_req > 0 else 0

        # 4. Education matching features
        edu_levels = {
            'not specified': 0,
            'bachelor': 1, 'b.sc': 1, 'b.a': 1, 'b.tech': 1,
            'master': 2, 'm.sc': 2, 'm.a': 2, 'm.tech': 2, 'mba': 2,
            'phd': 3, 'doctorate': 3
        }

        cv_edu_level = 0
        for edu in cv_data['education']:
            degree = edu['degree'].lower()
            for key, level in edu_levels.items():
                if key in degree:
                    cv_edu_level = max(cv_edu_level, level)

        jd_edu_level = 0
        jd_edu_req = jd_data['education_required'].lower()
        for key, level in edu_levels.items():
            if key in jd_edu_req:
                jd_edu_level = level
                break

        features['cv_education_level'] = cv_edu_level
        features['jd_education_level'] = jd_edu_level
        features['education_match'] = 1 if cv_edu_level >= jd_edu_level else cv_edu_level / jd_edu_level if jd_edu_level > 0 else 0

        # 5. Additional features
        features['cv_length'] = len(cv_text.split())
        features['jd_length'] = len(jd_text.split())

        return features

    def create_training_data(self) -> Tuple[List[Dict], List[Dict], List[float]]:
        """
        Create synthetic training data for model development.

        Returns:
            Tuple of (cv_data_list, jd_data_list, scores)
            Generates 100 realistic CV-job pairs with scores

        Note: In production, replace with real historical data
        """
        cv_processor = SimpleCVProcessor()
        jd_processor = SimpleJobProcessor()

        cv_data_list = []
        jd_data_list = []
        scores = []

        # Define realistic job roles and requirements
        job_roles = [
            {
                'title': 'Data Scientist',
                'required_skills': ['python', 'machine learning', 'sql', 'statistics', 'pandas', 'tensorflow'],
                'exp_req': 3,
                'edu_req': 'Master'
            },
            {
                'title': 'Software Engineer',
                'required_skills': ['python', 'java', 'django', 'react', 'aws', 'docker'],
                'exp_req': 2,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'DevOps Engineer',
                'required_skills': ['docker', 'kubernetes', 'aws', 'ci/cd', 'linux', 'terraform'],
                'exp_req': 3,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'Frontend Developer',
                'required_skills': ['javascript', 'react', 'html', 'css', 'typescript', 'node.js'],
                'exp_req': 2,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'Backend Developer',
                'required_skills': ['python', 'django', 'postgresql', 'rest apis', 'docker', 'aws'],
                'exp_req': 3,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'ML Engineer',
                'required_skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'aws', 'docker'],
                'exp_req': 4,
                'edu_req': 'Master'
            },
            {
                'title': 'Data Analyst',
                'required_skills': ['sql', 'python', 'excel', 'tableau', 'statistics', 'pandas'],
                'exp_req': 2,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'Full Stack Developer',
                'required_skills': ['python', 'javascript', 'react', 'django', 'sql', 'aws'],
                'exp_req': 3,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'Cloud Engineer',
                'required_skills': ['aws', 'azure', 'docker', 'kubernetes', 'terraform', 'linux'],
                'exp_req': 3,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'Senior Data Scientist',
                'required_skills': ['python', 'machine learning', 'deep learning', 'sql', 'tensorflow', 'aws'],
                'exp_req': 5,
                'edu_req': 'PhD'
            }
        ]

        # Generate 100 training samples with realistic variations
        np.random.seed(42)
        for i in range(100):
            role_idx = i % len(job_roles)
            role = job_roles[role_idx]

            # Generate candidate profile
            candidate_exp = np.random.randint(1, 10)
            candidate_edu_level = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            edu_map = {1: 'Bachelor', 2: 'Master', 3: 'PhD'}
            candidate_edu = edu_map[candidate_edu_level]

            # Generate skills with 70% match rate
            candidate_skills = []
            for skill in role['required_skills']:
                if np.random.random() > 0.3:
                    candidate_skills.append(skill)

            # Add random extra skills
            all_skills = []
            for skill_list in cv_processor.skill_database.values():
                all_skills.extend(skill_list)

            extra_skills = np.random.choice(all_skills, size=np.random.randint(0, 5), replace=False)
            candidate_skills.extend(extra_skills)

            # Create realistic CV text
            cv_text = f"""
            Candidate {i+1}

            Summary:
            Experienced professional with {candidate_exp} years in {role['title'].lower()}.

            Skills:
            {', '.join(candidate_skills)}

            Experience:
            {candidate_exp} years as {role['title']} at various companies.

            Education:
            {candidate_edu} in Computer Science
            """

            # Create realistic job description
            jd_text = f"""
            {role['title']}

            Requirements:
            - {role['exp_req']}+ years experience
            - {role['edu_req']} degree in Computer Science or related field
            - Strong skills in {', '.join(role['required_skills'][:3])}

            Responsibilities:
            - Develop and maintain solutions
            - Collaborate with team members
            - Write clean and efficient code

            Required Skills:
            {', '.join(role['required_skills'])}
            """

            # Calculate realistic ground truth score
            skill_match = len(set(candidate_skills) & set(role['required_skills'])) / len(role['required_skills'])
            exp_match = min(1, candidate_exp / role['exp_req'])
            edu_match = 1 if candidate_edu_level >= {'Bachelor': 1, 'Master': 2, 'PhD': 3}[role['edu_req']] else 0.5

            base_score = (skill_match * 0.5 + exp_match * 0.3 + edu_match * 0.2) * 100
            noise = np.random.normal(0, 5)  # Add realistic noise
            final_score = max(0, min(100, base_score + noise))

            # Parse and store data
            cv_data = cv_processor.parse_cv_text(cv_text)
            jd_data = jd_processor.parse_job_description(jd_text)

            cv_data_list.append(cv_data)
            jd_data_list.append(jd_data)
            scores.append(final_score)

        return cv_data_list, jd_data_list, scores

    def train(self, use_sample_data: bool = True):
        """
        Train the Random Forest model.

        Args:
            use_sample_data: Whether to use generated sample data (True for demo)

        Training Process:
            1. Create/load training data
            2. Extract features
            3. Split into train/test (80/20)
            4. Scale features
            5. Train Random Forest model
            6. Evaluate performance
        """
        if use_sample_data:
            print("[INFO] Creating sample training data...")
            cv_data_list, jd_data_list, scores = self.create_training_data()
        else:
            # For production: Load your actual data here
            pass

        print(f"[INFO] Training with {len(cv_data_list)} samples...")

        # Extract features for all CV-job pairs
        features_list = []
        for cv_data, jd_data in zip(cv_data_list, jd_data_list):
            features = self.extract_features(cv_data, jd_data)
            features_list.append(list(features.values()))

        X = np.array(features_list)
        y = np.array(scores)

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features for better model performance
        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Train Random Forest Regressor
        print("[INFO] Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,      # 100 decision trees
            max_depth=10,          # Maximum depth of trees
            min_samples_split=5,   # Minimum samples to split node
            random_state=42,       # For reproducibility
            n_jobs=-1              # Use all CPU cores
        )

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate model performance
        y_pred = self.model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display performance metrics
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R² Score (Variance Explained): {r2:.3f}")
        print("="*60)

        # Display top important features
        if hasattr(self.model, 'feature_importances_'):
            print("\nTOP 5 IMPORTANT FEATURES:")
            feature_names = list(self.extract_features(cv_data_list[0], jd_data_list[0]).keys())
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]

            for idx in top_indices:
                if idx < len(feature_names):
                    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

        print("\n[SUCCESS] Model training completed!")

    def predict(self, cv_data: Dict, jd_data: Dict, use_ml: bool = True) -> Dict:
        """
        Predict compatibility score for a CV-job pair.

        Args:
            cv_data: Parsed CV data
            jd_data: Parsed job description data
            use_ml: Whether to use ML model or rule-based scoring

        Returns:
            Dictionary containing:
                - fit_score: Predicted score (0-100%)
                - method: 'ML Model' or 'Rule-based'
                - features: Extracted feature values
                - analysis: Detailed breakdown and recommendations
        """
        if not self.is_trained and use_ml:
            print("[WARNING] ML model not trained. Switching to rule-based scoring...")
            use_ml = False

        # Extract features
        features = self.extract_features(cv_data, jd_data)

        if use_ml:
            # Scale features and predict using ML model
            feature_values = np.array([list(features.values())])
            feature_values_scaled = self.feature_scaler.transform(feature_values)

            score = self.model.predict(feature_values_scaled)[0]
            score = max(0, min(100, score))  # Ensure score is between 0-100
            method = "ML Model"
        else:
            # Fallback to rule-based scoring
            score = self._rule_based_scoring(features)
            method = "Rule-based"

        return {
            'fit_score': round(score, 1),
            'method': method,
            'features': features,
            'analysis': self._generate_analysis(cv_data, jd_data, features, score)
        }

    def _rule_based_scoring(self, features: Dict) -> float:
        """
        Rule-based scoring as fallback when ML model is not available.

        Args:
            features: Extracted feature values

        Returns:
            Score between 0-100 based on weighted features
        """
        weights = {
            'skill_match_ratio': 0.5,       # 50% weight on skill match
            'experience_sufficiency': 0.3,   # 30% weight on experience
            'education_match': 0.2           # 20% weight on education
        }

        score = 0
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight * 100

        return min(100, score)

    def _generate_analysis(self, cv_data: Dict, jd_data: Dict, features: Dict, score: float) -> Dict:
        """
        Generate detailed analysis of the prediction.

        Args:
            cv_data: Parsed CV data
            jd_data: Parsed job description data
            features: Extracted feature values
            score: Predicted fit score

        Returns:
            Dictionary with detailed analysis and recommendations
        """
        # Analyze skill matching
        cv_skills_flat = []
        for category, skills in cv_data['skills'].items():
            cv_skills_flat.extend(skills)

        jd_skills = jd_data['skills']
        matched_skills = set(cv_skills_flat) & set(jd_skills)
        missing_skills = set(jd_skills) - set(cv_skills_flat)

        # Interpret score
        if score >= 85:
            interpretation = "🎯 Excellent match - Highly recommended for interview"
        elif score >= 70:
            interpretation = "👍 Good match - Strong candidate worth considering"
        elif score >= 60:
            interpretation = "🤔 Fair match - Consider with additional screening"
        elif score >= 50:
            interpretation = "⚠️ Basic match - Might need additional training"
        else:
            interpretation = "❌ Poor match - Not recommended for this role"

        analysis = {
            'interpretation': interpretation,
            'skill_analysis': {
                'matched_skills': list(matched_skills)[:10],
                'missing_skills': list(missing_skills)[:10],
                'match_percentage': round(features.get('skill_match_ratio', 0) * 100, 1),
                'total_matched': len(matched_skills),
                'total_missing': len(missing_skills)
            },
            'experience_analysis': {
                'candidate_experience': features.get('cv_experience', 0),
                'required_experience': features.get('jd_required_experience', 0),
                'gap': features.get('experience_gap', 0),
                'sufficiency': "✅ Meets requirement" if features.get('experience_gap', 0) >= 0
                              else f"❌ Short by {abs(features.get('experience_gap', 0)):.1f} years"
            },
            'education_analysis': {
                'candidate_education': [f"{edu['degree']} in {edu['field']}" for edu in cv_data['education']]
                                      if cv_data['education'] else ["Not specified"],
                'required_education': jd_data['education_required'],
                'match_level': "✅ Meets or exceeds" if features.get('education_match', 0) == 1
                              else "❌ Below requirement"
            }
        }

        # Generate actionable recommendations
        recommendations = []

        if features.get('skill_match_ratio', 0) < 0.6:
            recommendations.append("📚 Consider acquiring some of the missing skills through online courses (Coursera, Udemy) or personal projects")

        if features.get('experience_gap', 0) < -1:
            recommendations.append(f"⏳ Gain {abs(features.get('experience_gap', 0)):.1f} more years of relevant experience through internships, freelance work, or internal projects")

        if features.get('education_match', 0) < 1:
            recommendations.append("🎓 Consider additional education, certifications, or specialized courses in the required field")

        if len(matched_skills) < 3:
            recommendations.append("💡 Highlight your core skills more prominently in your CV's summary and skills sections")

        if features.get('text_similarity', 0) < 0.3:
            recommendations.append("📝 Tailor your CV language to better match the job description keywords and terminology")

        analysis['recommendations'] = recommendations

        return analysis

    def save_model(self, filename: str = "candidate_fit_model.pkl"):
        """
        Save trained model and components to file.

        Args:
            filename: Path to save the model

        Note: Uses joblib for efficient serialization
        """
        import joblib

        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filename)
        print(f"[INFO] Model saved to {filename}")

    def load_model(self, filename: str = "candidate_fit_model.pkl"):
        """
        Load trained model from file.

        Args:
            filename: Path to saved model file
        """
        import joblib

        model_data = joblib.load(filename)

        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.is_trained = model_data['is_trained']

        print(f"[INFO] Model loaded from {filename}")


def display_banner():
    """Display ASCII art banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🎯 CANDIDATE JOB FIT PREDICTOR 🎯                   ║
    ║               Machine Learning System                        ║
    ║                                                              ║
    ║    Analyze CVs against job descriptions with AI accuracy    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def get_user_input() -> Tuple[str, str]:
    """
    Get CV and job description input from user.

    Returns:
        Tuple of (cv_text, jd_text)
    """
    print("\n" + "="*60)
    print("INPUT OPTIONS")
    print("="*60)

    use_sample = input("\nUse sample data for quick demo? (y/n): ").lower()

    if use_sample == 'y':
        # Realistic sample data
        cv_text = """
        Alexandra Chen
        Senior Data Scientist

        Summary:
        Experienced Data Scientist with 5 years in machine learning, data analysis,
        and predictive modeling. Strong background in Python, TensorFlow, and cloud technologies.

        Skills:
        Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL,
        AWS, Docker, Kubernetes, Pandas, NumPy, Scikit-learn, Statistics,
        Data Visualization, Tableau, Big Data, Spark

        Experience:
        5 years as Senior Data Scientist at TechInnovate Inc.
        2 years as Data Analyst at DataCorp LLC.

        Education:
        Master of Science in Computer Science - Stanford University
        Bachelor of Science in Statistics - University of California
        """

        jd_text = """
        Senior Data Scientist - AI Research Team

        Job Description:
        We're looking for an experienced Senior Data Scientist to join our AI Research team.

        Requirements:
        - 4+ years experience in data science or machine learning roles
        - Strong proficiency in Python and machine learning libraries (TensorFlow, PyTorch)
        - Experience with cloud platforms (AWS, GCP, or Azure)
        - Solid understanding of statistics and data analysis
        - Master's degree in Computer Science, Statistics, or related field

        Responsibilities:
        - Develop and deploy machine learning models in production
        - Conduct research on new AI algorithms and techniques
        - Collaborate with engineering teams to implement solutions
        - Analyze large datasets and create data visualizations
        - Mentor junior data scientists

        Required Skills:
        Python, Machine Learning, TensorFlow, SQL, AWS, Statistics,
        Data Analysis, Deep Learning, Cloud Computing

        Preferred Qualifications:
        - PhD in relevant field
        - Experience with Docker and Kubernetes
        - Published research in AI/ML conferences
        """

        print("\n[INFO] Using realistic sample data...")

    else:
        # Get custom input from user
        print("\n📄 ENTER YOUR CV/RESUME")
        print("="*40)
        print("Include: Skills, Experience, Education")
        print("Press Enter twice when finished.\n")

        cv_lines = []
        while True:
            line = input()
            if line == "" and cv_lines:
                if len(cv_lines) > 1 and cv_lines[-1] == "":
                    cv_lines.pop()
                    break
            cv_lines.append(line)

        cv_text = "\n".join(cv_lines)

        print("\n📋 ENTER JOB DESCRIPTION")
        print("="*40)
        print("Include: Requirements, Responsibilities, Required Skills")
        print("Press Enter twice when finished.\n")

        jd_lines = []
        while True:
            line = input()
            if line == "" and jd_lines:
                if len(jd_lines) > 1 and jd_lines[-1] == "":
                    jd_lines.pop()
                    break
            jd_lines.append(line)

        jd_text = "\n".join(jd_lines)

    return cv_text, jd_text


def display_results(result: Dict, cv_data: Dict, jd_data: Dict):
    """
    Display prediction results in user-friendly format.

    Args:
        result: Prediction result dictionary
        cv_data: Parsed CV data
        jd_data: Parsed job description data
    """
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    score = result['fit_score']
    method = result.get('method', 'Rule-based')

    # Color-coded score display
    if score >= 85:
        score_display = f"\033[92m{score}% (Excellent)\033[0m"
    elif score >= 70:
        score_display = f"\033[93m{score}% (Good)\033[0m"
    elif score >= 50:
        score_display = f"\033[33m{score}% (Fair)\033[0m"
    else:
        score_display = f"\033[91m{score}% (Poor)\033[0m"

    print(f"\n🏆 FINAL FIT SCORE: {score_display}")
    print(f"🔧 Prediction Method: {method}")

    analysis = result['analysis']

    print(f"\n📊 INTERPRETATION: {analysis['interpretation']}")

    print("\n" + "="*60)
    print("DETAILED ANALYSIS BREAKDOWN")
    print("="*60)

    # Candidate Information
    print(f"\n👤 CANDIDATE PROFILE")
    print(f"   Name: {cv_data.get('name', 'Unknown')}")
    print(f"   Total Skills: {cv_data.get('total_cv_skills', sum(len(s) for s in cv_data['skills'].values()))}")
    print(f"   Experience: {cv_data.get('experience', 0):.1f} years")
    if cv_data.get('education'):
        print(f"   Education: {', '.join([f'{edu['degree']} in {edu['field']}' for edu in cv_data['education']])}")

    # Job Information
    print(f"\n💼 JOB REQUIREMENTS")
    print(f"   Title: {jd_data.get('title', 'Unknown')}")
    print(f"   Required Experience: {jd_data.get('experience_required', 0):.1f} years")
    print(f"   Required Education: {jd_data.get('education_required', 'Not specified')}")
    print(f"   Required Skills: {len(jd_data.get('skills', []))}")

    # Skills Analysis
    skill_analysis = analysis['skill_analysis']
    print(f"\n🛠️  SKILLS ANALYSIS")
    print(f"   Match Rate: {skill_analysis['match_percentage']}%")
    print(f"   Matched: {skill_analysis['total_matched']} skills")
    print(f"   Missing: {skill_analysis['total_missing']} skills")

    if skill_analysis['matched_skills']:
        print(f"\n   ✅ STRENGTHS (Matched Skills):")
        for i, skill in enumerate(skill_analysis['matched_skills'][:5], 1):
            print(f"      {i}. {skill}")

    if skill_analysis['missing_skills']:
        print(f"\n   📝 AREAS FOR IMPROVEMENT (Missing Skills):")
        for i, skill in enumerate(skill_analysis['missing_skills'][:5], 1):
            print(f"      {i}. {skill}")

    # Experience Analysis
    exp_analysis = analysis['experience_analysis']
    print(f"\n⏳ EXPERIENCE ANALYSIS")
    print(f"   Candidate: {exp_analysis['candidate_experience']:.1f} years")
    print(f"   Required: {exp_analysis['required_experience']:.1f} years")
    print(f"   Gap: {exp_analysis['gap']:+.1f} years")
    print(f"   Status: {exp_analysis['sufficiency']}")

    # Education Analysis
    edu_analysis = analysis['education_analysis']
    print(f"\n🎓 EDUCATION ANALYSIS")
    print(f"   Candidate: {', '.join(edu_analysis['candidate_education'])}")
    print(f"   Required: {edu_analysis['required_education']}")
    print(f"   Status: {edu_analysis['match_level']}")

    # Key Metrics
    print(f"\n📈 KEY METRICS")
    features = result['features']
    metrics = [
        ('Skill Match Ratio', features.get('skill_match_ratio', 0) * 100, '%'),
        ('Text Similarity', features.get('text_similarity', 0) * 100, '%'),
        ('Experience Sufficiency', features.get('experience_sufficiency', 0) * 100, '%'),
        ('Education Match', features.get('education_match', 0) * 100, '%')
    ]

    for name, value, unit in metrics:
        print(f"   {name}: {value:.1f}{unit}")

    # Recommendations
    if analysis['recommendations']:
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")

    # Top Factors
    print(f"\n🏆 TOP 3 MATCHING FACTORS")
    features = result['features']
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    for i, (feature, value) in enumerate(sorted_features, 1):
        feature_name = feature.replace('_', ' ').title()
        if 'ratio' in feature or 'match' in feature or 'similarity' in feature:
            display_value = f"{value*100:.1f}%"
        elif 'experience' in feature or 'length' in feature or 'count' in feature:
            display_value = f"{value:.0f}"
        else:
            display_value = f"{value:.3f}"
        print(f"   {i}. {feature_name}: {display_value}")


def main():
    """
    Main execution function for the Candidate Fit Predictor.

    Flow:
    1. Display banner and initialize
    2. Train ML model (or load existing)
    3. Get user input (CV and job description)
    4. Process inputs and extract features
    5. Predict compatibility score
    6. Display detailed results
    7. Save report (optional)
    """
    # Display welcome banner
    display_banner()

    # Initialize predictor
    print("[INFO] Initializing Candidate Fit Predictor...")
    predictor = CandidateFitPredictor()

    # Train or load model
    print("\n" + "="*60)
    print("MODEL SETUP")
    print("="*60)

    try:
        # Try to load existing model
        predictor.load_model("candidate_fit_model.pkl")
        print("[INFO] Loaded pre-trained model")
    except:
        # Train new model
        print("[INFO] Training new model...")
        predictor.train(use_sample_data=True)
        predictor.save_model("candidate_fit_model.pkl")

    # Get user input
    cv_text, jd_text = get_user_input()

    # Process inputs
    print("\n" + "="*60)
    print("PROCESSING INPUTS")
    print("="*60)

    cv_processor = SimpleCVProcessor()
    jd_processor = SimpleJobProcessor()

    cv_data = cv_processor.parse_cv_text(cv_text)
    jd_data = jd_processor.parse_job_description(jd_text)

    print(f"[SUCCESS] CV processed successfully")
    print(f"   Name: {cv_data.get('name', 'Unknown')}")
    print(f"   Skills identified: {sum(len(s) for s in cv_data['skills'].values())}")
    print(f"   Experience: {cv_data.get('experience', 0):.1f} years")

    print(f"\n[SUCCESS] Job description processed successfully")
    print(f"   Title: {jd_data.get('title', 'Unknown')}")
    print(f"   Required skills: {len(jd_data.get('skills', []))}")
    print(f"   Required experience: {jd_data.get('experience_required', 0):.1f} years")

    # Make prediction
    print("\n" + "="*60)
    print("ANALYZING COMPATIBILITY")
    print("="*60)

    result = predictor.predict(cv_data, jd_data, use_ml=True)

    # Display results
    display_results(result, cv_data, jd_data)

    # Optional: Save report
    print("\n" + "="*60)
    save_report = input("\n💾 Save detailed report to file? (y/n): ").lower()

    if save_report == 'y':
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fit_report_{timestamp}.json"

        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'candidate': {
                'name': cv_data.get('name', 'Unknown'),
                'experience_years': cv_data.get('experience', 0),
                'education': cv_data.get('education', [])
            },
            'job': {
                'title': jd_data.get('title', 'Unknown'),
                'required_experience': jd_data.get('experience_required', 0),
                'required_education': jd_data.get('education_required', 'Not specified')
            },
            'prediction': result
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Report saved to {filename}")

    # Final message
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nThank you for using Candidate Fit Predictor! 💼✨")
    print("\nFor more accurate results:")
    print("1. Provide detailed CV with specific skills and experience")
    print("2. Include complete job description with requirements")
    print("3. For production use, train with your historical hiring data")


# Run the application
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("[INFO] Please check your inputs and try again.")