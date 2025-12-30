"""
Candidate Fit Score Predictor
=============================

A machine learning system that analyzes candidate CVs/resumes against job descriptions
to predict compatibility scores (0-100%) with detailed explanations.

Author: AI Assistant
Date: 2024
Version: 2.0
License: MIT

Features:
- Text similarity analysis using TF-IDF and cosine similarity
- Skill matching with comprehensive skill database
- Experience and education level matching
- ML model (Random Forest) for accurate predictions
- Detailed analysis with recommendations
- Model persistence and loading
- Comprehensive error handling

Requirements:
- Python 3.8+
- scikit-learn, pandas, numpy, nltk
- joblib for model serialization

Usage:
    python candidate_fit_predictor.py
    OR
    from candidate_fit_predictor import CandidateFitPredictor

API Example:
    >>> predictor = CandidateFitPredictor()
    >>> score, analysis = predictor.predict_cv_job(cv_text, jd_text)
    >>> print(f"Fit Score: {score}%")
"""

import pandas as pd
import numpy as np
import re
import json
import warnings
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
warnings.filterwarnings('ignore')

# Basic data processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML models
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except:
            nltk.download(resource, quiet=True)

download_nltk_resources()


@dataclass
class CandidateProfile:
    """Data class for candidate profile."""
    name: str
    skills: Dict[str, List[str]]
    experience: float
    education: List[Dict[str, str]]
    raw_text: str
    processed_features: Optional[Dict] = None


@dataclass
class JobDescription:
    """Data class for job description."""
    title: str
    required_skills: List[str]
    required_experience: float
    required_education: str
    raw_text: str


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove special chars, lemmatize.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)


class SkillDatabase:
    """Comprehensive skill database with categorization."""
    
    def __init__(self):
        self.skills = {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
                'swift', 'kotlin', 'go', 'rust', 'typescript', 'scala', 'r',
                'matlab', 'perl', 'bash', 'shell', 'html', 'css'
            ],
            'frameworks': [
                'django', 'flask', 'spring', 'express', 'laravel', 'asp.net',
                'ruby on rails', 'react', 'angular', 'vue', 'node.js',
                'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'spark'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
                'sqlite', 'cassandra', 'dynamodb', 'elasticsearch',
                'firebase', 'mariadb'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                'ansible', 'jenkins', 'git', 'ci/cd', 'devops', 'linux', 'unix'
            ],
            'data_ai': [
                'machine learning', 'deep learning', 'artificial intelligence',
                'nlp', 'computer vision', 'data science', 'data analysis',
                'statistics', 'pandas', 'numpy', 'matplotlib', 'seaborn',
                'tableau', 'power bi', 'excel', 'hadoop'
            ],
            'soft_skills': [
                'communication', 'leadership', 'teamwork', 'problem solving',
                'critical thinking', 'adaptability', 'time management',
                'creativity', 'collaboration', 'project management',
                'agile', 'scrum'
            ]
        }
        
        # Create reverse mapping for quick lookup
        self.skill_to_category = {}
        for category, skill_list in self.skills.items():
            for skill in skill_list:
                self.skill_to_category[skill] = category
    
    def get_skill_category(self, skill: str) -> str:
        """Get category for a given skill."""
        return self.skill_to_category.get(skill.lower(), 'other')
    
    def get_all_skills(self) -> List[str]:
        """Get all skills as a flat list."""
        all_skills = []
        for skill_list in self.skills.values():
            all_skills.extend(skill_list)
        return list(set(all_skills))


class CVProcessor:
    """
    Advanced CV Processor for extracting information from CV/resume text.
    
    Features:
    - Skill extraction with confidence scoring
    - Experience parsing with job role detection
    - Education level normalization
    - Multi-format CV support (text-based)
    """
    
    def __init__(self, skill_db: Optional[SkillDatabase] = None):
        self.skill_db = skill_db or SkillDatabase()
        self.preprocessor = TextPreprocessor()
        
    def parse_cv(self, cv_text: str) -> CandidateProfile:
        """
        Parse CV text into structured CandidateProfile.
        
        Args:
            cv_text: Raw CV text content
            
        Returns:
            CandidateProfile object
        """
        logger.info("Processing CV text...")
        
        # Extract components
        name = self._extract_name(cv_text)
        skills = self._extract_skills_with_context(cv_text)
        experience = self._extract_experience_detailed(cv_text)
        education = self._extract_education_detailed(cv_text)
        
        return CandidateProfile(
            name=name,
            skills=skills,
            experience=experience['total_years'],
            education=education,
            raw_text=cv_text.lower()
        )
    
    def _extract_name(self, text: str) -> str:
        """Extract candidate name using multiple heuristics."""
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a name (2-4 words, capitalized)
            words = first_line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                return first_line
        
        # Fallback: look for patterns like "Name:", "Candidate:", etc.
        patterns = [
            r'name:\s*([A-Za-z\s]+)',
            r'candidate:\s*([A-Za-z\s]+)',
            r'applicant:\s*([A-Za-z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return "Unknown Candidate"
    
    def _extract_skills_with_context(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills with context awareness and confidence.
        
        Returns:
            Dict with categories as keys and lists of skills as values
        """
        found_skills = {category: [] for category in self.skill_db.skills.keys()}
        found_skills['other'] = []
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Look for skills in each category
        for category, skill_list in self.skill_db.skills.items():
            for skill in skill_list:
                # Check for skill with word boundaries
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, processed_text, re.IGNORECASE):
                    found_skills[category].append(skill)
        
        # Look for skill sections
        skill_section_patterns = [
            r'skills?(?:\s*&?\s*qualifications?)?:?(.*?)(?:\n\n|\n[A-Z]|$)',
            r'technical\s+skills?:?(.*?)(?:\n\n|\n[A-Z]|$)',
            r'competencies?:?(.*?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in skill_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1)
                for skill in self.skill_db.get_all_skills():
                    if re.search(r'\b' + re.escape(skill) + r'\b', section_text, re.IGNORECASE):
                        category = self.skill_db.get_skill_category(skill)
                        if skill not in found_skills[category]:
                            found_skills[category].append(skill)
        
        return {k: v for k, v in found_skills.items() if v}
    
    def _extract_experience_detailed(self, text: str) -> Dict[str, Any]:
        """
        Extract detailed experience information.
        
        Returns:
            Dict with total_years, roles, and companies
        """
        total_years = 0.0
        roles = []
        
        # Pattern for years of experience
        patterns = [
            r'(\d+)\+?\s*(?:year|yr|years|yrs)(?:\s+of)?\s+experience',
            r'experience:\s*(\d+)\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?\s*experience',
            r'(\d+)\s*years?\s*(?:in|of)\s*.*?\s*experience'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    years = [g for g in match.groups() if g and g.isdigit()]
                    if years:
                        total_years = max(total_years, float(max(years)))
        
        # Extract job roles and durations
        role_pattern = r'(\d+(?:\.\d+)?)\s*years?\s*(?:as|in)\s*([A-Za-z\s]+?)(?:,|\.|\n|$)'
        for match in re.finditer(role_pattern, text, re.IGNORECASE):
            years = float(match.group(1))
            role = match.group(2).strip()
            roles.append({'role': role, 'years': years})
        
        return {
            'total_years': total_years,
            'roles': roles,
            'estimated': len(roles) == 0  # Flag if estimated
        }
    
    def _extract_education_detailed(self, text: str) -> List[Dict[str, str]]:
        """Extract detailed education information."""
        education = []
        
        patterns = [
            r'(bachelor|b\.?s\.?c?|b\.?a\.?|b\.?tech)\s*(?:in|of)?\s*([A-Za-z\s&]+?)(?:,|\.|\n|$)',
            r'(master|m\.?s\.?c?|m\.?a\.?|m\.?tech|mba)\s*(?:in|of)?\s*([A-Za-z\s&]+?)(?:,|\.|\n|$)',
            r'(ph\.?d\.?|doctorate)\s*(?:in|of)?\s*([A-Za-z\s&]+?)(?:,|\.|\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                degree = self._normalize_degree(match.group(1))
                field = match.group(2).strip().title() if match.group(2) else 'Not specified'
                education.append({
                    'degree': degree,
                    'field': field,
                    'level': self._get_education_level(degree)
                })
        
        return education
    
    def _normalize_degree(self, degree: str) -> str:
        """Normalize degree names."""
        degree_lower = degree.lower()
        if 'phd' in degree_lower or 'doctorate' in degree_lower:
            return 'PhD'
        elif 'master' in degree_lower or 'mba' in degree_lower:
            return "Master's"
        elif 'bachelor' in degree_lower or 'bs' in degree_lower or 'ba' in degree_lower:
            return "Bachelor's"
        else:
            return degree.title()
    
    def _get_education_level(self, degree: str) -> int:
        """Convert degree to numeric level (0-3)."""
        levels = {
            "high school": 0,
            "associate": 1,
            "bachelor's": 2,
            "master's": 3,
            "mba": 3,
            "phd": 4
        }
        return levels.get(degree.lower(), 0)


class JobDescriptionProcessor:
    """
    Processor for extracting requirements from job descriptions.
    """
    
    def __init__(self, skill_db: Optional[SkillDatabase] = None):
        self.skill_db = skill_db or SkillDatabase()
        self.preprocessor = TextPreprocessor()
    
    def parse_job_description(self, jd_text: str) -> JobDescription:
        """
        Parse job description text.
        
        Args:
            jd_text: Raw job description text
            
        Returns:
            JobDescription object
        """
        logger.info("Processing job description...")
        
        # Extract components
        title = self._extract_title(jd_text)
        required_skills = self._extract_required_skills(jd_text)
        required_experience = self._extract_required_experience(jd_text)
        required_education = self._extract_required_education(jd_text)
        
        return JobDescription(
            title=title,
            required_skills=required_skills,
            required_experience=required_experience,
            required_education=required_education,
            raw_text=jd_text.lower()
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract job title."""
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a title
            if len(first_line) < 100 and not first_line.islower():
                return first_line.title()
        
        # Look for title patterns
        patterns = [
            r'position:\s*(.+?)(?:\n|$)',
            r'role:\s*(.+?)(?:\n|$)',
            r'job\s+title:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return "Unknown Position"
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required skills from job description."""
        skills = []
        
        # Look for skills sections
        section_patterns = [
            r'requirements?:?(.*?)(?:\n\n|\n[A-Z]|$)',
            r'required\s+skills?:?(.*?)(?:\n\n|\n[A-Z]|$)',
            r'qualifications?:?(.*?)(?:\n\n|\n[A-Z]|$)',
            r'must\s+have:?(.*?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1)
                # Check for skills in this section
                for skill in self.skill_db.get_all_skills():
                    if re.search(r'\b' + re.escape(skill) + r'\b', section_text, re.IGNORECASE):
                        if skill not in skills:
                            skills.append(skill)
        
        # Also check whole text for skills
        processed_text = self.preprocessor.preprocess_text(text)
        for skill in self.skill_db.get_all_skills():
            if re.search(r'\b' + re.escape(skill) + r'\b', processed_text, re.IGNORECASE):
                if skill not in skills:
                    skills.append(skill)
        
        return skills
    
    def _extract_required_experience(self, text: str) -> float:
        """Extract required years of experience."""
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
    
    def _extract_required_education(self, text: str) -> str:
        """Extract required education level."""
        patterns = [
            r'(bachelor|b\.?s\.?|b\.?a\.?)\s*(?:degree|in)?',
            r'(master|m\.?s\.?|m\.?a\.?|mba)\s*(?:degree|in)?',
            r'(ph\.?d\.?|doctorate)\s*(?:degree|in)?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                degree = match.group(1).lower()
                if 'phd' in degree or 'doctorate' in degree:
                    return "PhD"
                elif 'master' in degree or 'mba' in degree:
                    return "Master's"
                elif 'bachelor' in degree:
                    return "Bachelor's"
        
        return "Not specified"


class FeatureExtractor:
    """
    Feature extraction for CV-job matching.
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
    
    def extract_features(self, cv: CandidateProfile, jd: JobDescription) -> Dict[str, float]:
        """
        Extract comprehensive feature set for ML model.
        
        Returns:
            Dict of feature names and values
        """
        features = {}
        
        # 1. Text similarity features
        features.update(self._extract_text_similarity(cv.raw_text, jd.raw_text))
        
        # 2. Skill matching features
        features.update(self._extract_skill_features(cv.skills, jd.required_skills))
        
        # 3. Experience features
        features.update(self._extract_experience_features(cv.experience, jd.required_experience))
        
        # 4. Education features
        features.update(self._extract_education_features(cv.education, jd.required_education))
        
        # 5. Additional features
        features.update(self._extract_additional_features(cv, jd))
        
        return features
    
    def _extract_text_similarity(self, cv_text: str, jd_text: str) -> Dict[str, float]:
        """Extract text similarity features."""
        features = {}
        
        try:
            # TF-IDF cosine similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([cv_text, jd_text])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            features['text_similarity'] = cosine_sim
            
            # Jaccard similarity on bag of words
            cv_words = set(cv_text.split())
            jd_words = set(jd_text.split())
            if cv_words and jd_words:
                jaccard_sim = len(cv_words & jd_words) / len(cv_words | jd_words)
                features['jaccard_similarity'] = jaccard_sim
            
        except Exception as e:
            logger.warning(f"Text similarity extraction failed: {e}")
            features['text_similarity'] = 0.0
            features['jaccard_similarity'] = 0.0
        
        return features
    
    def _extract_skill_features(self, cv_skills: Dict, jd_skills: List[str]) -> Dict[str, float]:
        """Extract skill matching features."""
        features = {}
        
        # Flatten CV skills
        cv_skills_flat = []
        for skill_list in cv_skills.values():
            cv_skills_flat.extend(skill_list)
        
        if jd_skills:
            matched_skills = set(cv_skills_flat) & set(jd_skills)
            
            features['skill_match_ratio'] = len(matched_skills) / len(jd_skills)
            features['matched_skills_count'] = len(matched_skills)
            features['total_jd_skills'] = len(jd_skills)
            
            # Category-based matching
            skill_db = SkillDatabase()
            for category in skill_db.skills.keys():
                cv_cat_skills = set(cv_skills.get(category, []))
                jd_cat_skills = set([s for s in jd_skills if skill_db.get_skill_category(s) == category])
                if jd_cat_skills:
                    features[f'{category}_match_ratio'] = len(cv_cat_skills & jd_cat_skills) / len(jd_cat_skills)
        else:
            features['skill_match_ratio'] = 0.0
            features['matched_skills_count'] = 0
            features['total_jd_skills'] = 0
        
        features['total_cv_skills'] = len(cv_skills_flat)
        
        return features
    
    def _extract_experience_features(self, cv_exp: float, jd_exp: float) -> Dict[str, float]:
        """Extract experience matching features."""
        features = {}
        
        features['cv_experience'] = cv_exp
        features['jd_required_experience'] = jd_exp
        features['experience_gap'] = cv_exp - jd_exp
        
        if jd_exp > 0:
            features['experience_sufficiency'] = min(1.0, cv_exp / jd_exp)
            features['experience_excess'] = max(0, cv_exp - jd_exp) / jd_exp
        else:
            features['experience_sufficiency'] = 1.0
            features['experience_excess'] = 0.0
        
        return features
    
    def _extract_education_features(self, cv_education: List[Dict], jd_education: str) -> Dict[str, float]:
        """Extract education matching features."""
        features = {}
        
        # Get highest education level from CV
        cv_levels = [edu.get('level', 0) for edu in cv_education]
        cv_highest_level = max(cv_levels) if cv_levels else 0
        
        # Convert JD education to level
        jd_level = self._education_to_level(jd_education)
        
        features['cv_education_level'] = cv_highest_level
        features['jd_education_level'] = jd_level
        features['education_gap'] = cv_highest_level - jd_level
        
        if jd_level > 0:
            features['education_match'] = 1.0 if cv_highest_level >= jd_level else cv_highest_level / jd_level
        else:
            features['education_match'] = 1.0
        
        return features
    
    def _education_to_level(self, education: str) -> int:
        """Convert education string to numeric level."""
        education_lower = education.lower()
        
        if 'phd' in education_lower or 'doctorate' in education_lower:
            return 4
        elif 'master' in education_lower or 'mba' in education_lower:
            return 3
        elif 'bachelor' in education_lower:
            return 2
        elif 'associate' in education_lower:
            return 1
        else:
            return 0
    
    def _extract_additional_features(self, cv: CandidateProfile, jd: JobDescription) -> Dict[str, float]:
        """Extract additional features."""
        features = {}
        
        # Text length features
        features['cv_length'] = len(cv.raw_text.split())
        features['jd_length'] = len(jd.raw_text.split())
        features['length_ratio'] = features['cv_length'] / max(1, features['jd_length'])
        
        # Skill diversity
        cv_skills_flat = []
        for skill_list in cv.skills.values():
            cv_skills_flat.extend(skill_list)
        
        features['skill_diversity'] = len(set(cv_skills_flat)) / max(1, len(cv_skills_flat))
        
        return features


class CandidateFitPredictor:
    """
    Main ML model for predicting candidate-job compatibility.
    
    Features:
    - Random Forest Regressor for score prediction
    - Feature importance analysis
    - Model persistence
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.cv_processor = CVProcessor()
        self.jd_processor = JobDescriptionProcessor()
        self.feature_extractor = FeatureExtractor()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def create_training_data(self, n_samples: int = 200) -> Tuple[List[Dict], List[float]]:
        """
        Create synthetic training data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_list, scores)
        """
        logger.info(f"Generating {n_samples} training samples...")
        
        features_list = []
        scores = []
        
        # Define realistic job roles
        job_roles = self._get_job_roles()
        
        np.random.seed(42)
        
        for i in range(n_samples):
            role = job_roles[i % len(job_roles)]
            
            # Generate candidate with realistic variations
            candidate_exp = np.random.uniform(role['exp_req'] - 2, role['exp_req'] + 3)
            candidate_exp = max(0.5, candidate_exp)
            
            # Generate education level
            edu_levels = {'Bachelor': 2, 'Master': 3, 'PhD': 4}
            candidate_edu_level = np.random.choice(
                list(edu_levels.values()),
                p=[0.6, 0.3, 0.1]
            )
            
            # Generate skills with realistic match rate
            match_rate = np.random.beta(2, 2)  # Beta distribution centered around 0.5
            candidate_skills = []
            for skill in role['required_skills']:
                if np.random.random() < match_rate:
                    candidate_skills.append(skill)
            
            # Add some extra skills
            skill_db = SkillDatabase()
            all_skills = skill_db.get_all_skills()
            extra_skills = np.random.choice(
                [s for s in all_skills if s not in role['required_skills']],
                size=np.random.randint(0, 8),
                replace=False
            )
            candidate_skills.extend(extra_skills)
            
            # Create CV and JD texts
            cv_text = self._generate_cv_text(i, candidate_skills, candidate_exp, candidate_edu_level)
            jd_text = self._generate_jd_text(role)
            
            # Parse and extract features
            cv_data = self.cv_processor.parse_cv(cv_text)
            jd_data = self.jd_processor.parse_job_description(jd_text)
            
            features = self.feature_extractor.extract_features(cv_data, jd_data)
            features_list.append(features)
            
            # Calculate ground truth score
            score = self._calculate_ground_truth_score(features)
            scores.append(score)
        
        return features_list, scores
    
    def _get_job_roles(self) -> List[Dict]:
        """Get predefined job roles."""
        return [
            {
                'title': 'Data Scientist',
                'required_skills': ['python', 'machine learning', 'sql', 'statistics', 'pandas'],
                'exp_req': 3,
                'edu_req': 'Master'
            },
            {
                'title': 'Software Engineer',
                'required_skills': ['python', 'java', 'django', 'aws', 'docker'],
                'exp_req': 2,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'DevOps Engineer',
                'required_skills': ['docker', 'kubernetes', 'aws', 'linux', 'terraform'],
                'exp_req': 3,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'Frontend Developer',
                'required_skills': ['javascript', 'react', 'html', 'css', 'typescript'],
                'exp_req': 2,
                'edu_req': 'Bachelor'
            },
            {
                'title': 'ML Engineer',
                'required_skills': ['python', 'machine learning', 'tensorflow', 'aws', 'docker'],
                'exp_req': 4,
                'edu_req': 'Master'
            },
            {
                'title': 'Data Analyst',
                'required_skills': ['sql', 'python', 'excel', 'tableau', 'statistics'],
                'exp_req': 2,
                'edu_req': 'Bachelor'
            }
        ]
    
    def _generate_cv_text(self, idx: int, skills: List[str], exp: float, edu_level: int) -> str:
        """Generate realistic CV text."""
        edu_map = {2: 'Bachelor of Science in Computer Science',
                  3: 'Master of Science in Data Science',
                  4: 'PhD in Computer Science'}
        
        return f"""
        Candidate {idx+1}
        
        Summary:
        Experienced professional with {exp:.1f} years of experience.
        
        Skills:
        {', '.join(skills)}
        
        Experience:
        {exp:.1f} years in various roles
        
        Education:
        {edu_map.get(edu_level, 'Bachelor of Science')}
        """
    
    def _generate_jd_text(self, role: Dict) -> str:
        """Generate realistic job description text."""
        return f"""
        {role['title']}
        
        Requirements:
        - {role['exp_req']}+ years experience
        - {role['edu_req']} degree required
        - Skills: {', '.join(role['required_skills'][:3])}
        
        Required Skills:
        {', '.join(role['required_skills'])}
        """
    
    def _calculate_ground_truth_score(self, features: Dict) -> float:
        """Calculate ground truth score for training data."""
        weights = {
            'skill_match_ratio': 0.4,
            'experience_sufficiency': 0.3,
            'education_match': 0.2,
            'text_similarity': 0.1
        }
        
        score = 0
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight * 100
        
        # Add some noise
        score += np.random.normal(0, 5)
        
        return max(0, min(100, score))
    
    def train(self, n_samples: int = 200) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Starting model training...")
        
        # Create training data
        features_list, scores = self.create_training_data(n_samples)
        
        # Convert to numpy arrays
        if self.feature_names is None:
            self.feature_names = list(features_list[0].keys())
        
        X = np.array([[features.get(name, 0) for name in self.feature_names] 
                      for features in features_list])
        y = np.array(scores)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.feature_extractor.scaler.fit_transform(X_train)
        X_test_scaled = self.feature_extractor.scaler.transform(X_test)
        
        # Train model
        logger.info("Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='r2'
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        # Display results
        self._display_training_results(metrics)
        
        return metrics
    
    def _display_training_results(self, metrics: Dict):
        """Display training results."""
        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)
        print(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"Cross-Validation R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
        
        if hasattr(self.model, 'feature_importances_'):
            print("\nTOP 10 FEATURE IMPORTANCES:")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-10:][::-1]
            
            for i, idx in enumerate(indices, 1):
                if idx < len(self.feature_names):
                    print(f"  {i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        print("="*60)
        print("[SUCCESS] Model training completed!")
    
    def predict(self, cv_text: str, jd_text: str) -> Tuple[float, Dict]:
        """
        Predict compatibility score.
        
        Args:
            cv_text: CV/resume text
            jd_text: Job description text
            
        Returns:
            Tuple of (score, analysis_dict)
        """
        # Parse inputs
        cv_data = self.cv_processor.parse_cv(cv_text)
        jd_data = self.jd_processor.parse_job_description(jd_text)
        
        # Extract features
        features = self.feature_extractor.extract_features(cv_data, jd_data)
        
        # Predict score
        if self.is_trained and self.feature_names:
            # Ensure features are in correct order
            feature_vector = np.array([[features.get(name, 0) for name in self.feature_names]])
            feature_vector_scaled = self.feature_extractor.scaler.transform(feature_vector)
            score = self.model.predict(feature_vector_scaled)[0]
            method = "ML Model"
        else:
            # Fallback to rule-based scoring
            score = self._rule_based_score(features)
            method = "Rule-based"
        
        # Generate analysis
        analysis = self._generate_analysis(cv_data, jd_data, features, score, method)
        
        return min(100, max(0, score)), analysis
    
    def _rule_based_score(self, features: Dict) -> float:
        """Rule-based scoring as fallback."""
        weights = {
            'skill_match_ratio': 0.5,
            'experience_sufficiency': 0.3,
            'education_match': 0.2
        }
        
        score = 0
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight * 100
        
        return score
    
    def _generate_analysis(self, cv: CandidateProfile, jd: JobDescription, 
                          features: Dict, score: float, method: str) -> Dict:
        """Generate detailed analysis."""
        # Skill analysis
        cv_skills_flat = []
        for skill_list in cv.skills.values():
            cv_skills_flat.extend(skill_list)
        
        matched_skills = set(cv_skills_flat) & set(jd.required_skills)
        missing_skills = set(jd.required_skills) - set(cv_skills_flat)
        
        # Score interpretation
        if score >= 85:
            interpretation = "🎯 Excellent match - Highly recommended for interview"
            recommendation = "Proceed to technical interview"
        elif score >= 70:
            interpretation = "👍 Good match - Strong candidate worth considering"
            recommendation = "Schedule initial screening"
        elif score >= 60:
            interpretation = "🤔 Fair match - Consider with additional screening"
            recommendation = "Review specific skill gaps"
        elif score >= 50:
            interpretation = "⚠️ Basic match - Might need additional training"
            recommendation = "Consider for junior role or with training plan"
        else:
            interpretation = "❌ Poor match - Not recommended for this role"
            recommendation = "Consider other candidates or different roles"
        
        # Generate recommendations
        recommendations = []
        
        if features.get('skill_match_ratio', 0) < 0.6:
            recommendations.append(
                f"Improve skill match: Consider acquiring {min(3, len(missing_skills))} "
                f"key missing skills: {', '.join(list(missing_skills)[:3])}"
            )
        
        if features.get('experience_gap', 0) < -1:
            gap = abs(features['experience_gap'])
            recommendations.append(
                f"Gain {gap:.1f} more years of relevant experience through "
                "projects or internships"
            )
        
        if features.get('education_match', 0) < 1:
            recommendations.append(
                "Consider relevant certifications or courses to supplement education"
            )
        
        if features.get('text_similarity', 0) < 0.3:
            recommendations.append(
                "Tailor CV language to better match job description keywords"
            )
        
        analysis = {
            'score': round(score, 1),
            'method': method,
            'interpretation': interpretation,
            'recommendation': recommendation,
            'candidate': {
                'name': cv.name,
                'experience_years': cv.experience,
                'education': cv.education,
                'skill_count': sum(len(s) for s in cv.skills.values())
            },
            'job': {
                'title': jd.title,
                'required_experience': jd.required_experience,
                'required_education': jd.required_education,
                'required_skill_count': len(jd.required_skills)
            },
            'skill_analysis': {
                'match_percentage': round(features.get('skill_match_ratio', 0) * 100, 1),
                'matched_skills': list(matched_skills)[:10],
                'missing_skills': list(missing_skills)[:10],
                'total_matched': len(matched_skills),
                'total_missing': len(missing_skills)
            },
            'experience_analysis': {
                'candidate': cv.experience,
                'required': jd.required_experience,
                'gap': features.get('experience_gap', 0),
                'sufficiency': "Meets requirement" if features.get('experience_gap', 0) >= 0
                              else f"Short by {abs(features.get('experience_gap', 0)):.1f} years"
            },
            'education_analysis': {
                'candidate': [f"{edu['degree']} in {edu['field']}" for edu in cv.education],
                'required': jd.required_education,
                'match': "Meets or exceeds" if features.get('education_match', 0) == 1
                        else "Below requirement"
            },
            'detailed_recommendations': recommendations,
            'key_metrics': {
                k: round(v, 3) for k, v in features.items() 
                if k in ['skill_match_ratio', 'text_similarity', 
                        'experience_sufficiency', 'education_match']
            }
        }
        
        return analysis
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        import joblib
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler': self.feature_extractor.scaler,
            'tfidf_vectorizer': self.feature_extractor.tfidf_vectorizer,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model
        """
        import joblib
        
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.feature_extractor.scaler = model_data['scaler']
            self.feature_extractor.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate(self, test_data: List[Tuple[str, str, float]]) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_data: List of (cv_text, jd_text, true_score)
            
        Returns:
            Evaluation metrics
        """
        predictions = []
        true_scores = []
        
        for cv_text, jd_text, true_score in test_data:
            pred_score, _ = self.predict(cv_text, jd_text)
            predictions.append(pred_score)
            true_scores.append(true_score)
        
        metrics = {
            'mse': mean_squared_error(true_scores, predictions),
            'mae': mean_absolute_error(true_scores, predictions),
            'r2': r2_score(true_scores, predictions),
            'rmse': np.sqrt(mean_squared_error(true_scores, predictions))
        }
        
        return metrics


def display_banner():
    """Display ASCII art banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🎯 CANDIDATE JOB FIT PREDICTOR 🎯                   ║
    ║               Version 2.0 - Production Ready                 ║
    ║                                                              ║
    ║    Analyze CVs against job descriptions with AI accuracy    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def display_results(score: float, analysis: Dict):
    """
    Display results in user-friendly format.
    
    Args:
        score: Predicted score
        analysis: Detailed analysis dictionary
    """
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    # Color-coded score
    if score >= 85:
        score_display = f"\033[92m{score}% (Excellent)\033[0m"
    elif score >= 70:
        score_display = f"\033[93m{score}% (Good)\033[0m"
    elif score >= 50:
        score_display = f"\033[33m{score}% (Fair)\033[0m"
    else:
        score_display = f"\033[91m{score}% (Poor)\033[0m"
    
    print(f"\n🏆 FINAL FIT SCORE: {score_display}")
    print(f"🔧 Method: {analysis['method']}")
    print(f"📊 Interpretation: {analysis['interpretation']}")
    print(f"💡 Recommendation: {analysis['recommendation']}")
    
    print(f"\n👤 CANDIDATE:")
    print(f"   Name: {analysis['candidate']['name']}")
    print(f"   Experience: {analysis['candidate']['experience_years']:.1f} years")
    print(f"   Education: {', '.join(analysis['candidate']['education'])}")
    print(f"   Skills: {analysis['candidate']['skill_count']} identified")
    
    print(f"\n💼 JOB:")
    print(f"   Title: {analysis['job']['title']}")
    print(f"   Required Experience: {analysis['job']['required_experience']:.1f} years")
    print(f"   Required Education: {analysis['job']['required_education']}")
    print(f"   Required Skills: {analysis['job']['required_skill_count']}")
    
    print(f"\n🛠️  SKILLS ANALYSIS:")
    print(f"   Match Rate: {analysis['skill_analysis']['match_percentage']}%")
    print(f"   Matched Skills: {analysis['skill_analysis']['total_matched']}")
    print(f"   Missing Skills: {analysis['skill_analysis']['total_missing']}")
    
    if analysis['skill_analysis']['matched_skills']:
        print(f"   ✅ Top Matched Skills:")
        for i, skill in enumerate(analysis['skill_analysis']['matched_skills'][:5], 1):
            print(f"      {i}. {skill}")
    
    if analysis['skill_analysis']['missing_skills']:
        print(f"   📝 Critical Missing Skills:")
        for i, skill in enumerate(analysis['skill_analysis']['missing_skills'][:5], 1):
            print(f"      {i}. {skill}")
    
    print(f"\n⏳ EXPERIENCE ANALYSIS:")
    print(f"   Candidate: {analysis['experience_analysis']['candidate']:.1f} years")
    print(f"   Required: {analysis['experience_analysis']['required']:.1f} years")
    print(f"   Gap: {analysis['experience_analysis']['gap']:+.1f} years")
    print(f"   Status: {analysis['experience_analysis']['sufficiency']}")
    
    print(f"\n🎓 EDUCATION ANALYSIS:")
    edu_candidate = analysis['education_analysis']['candidate']
    if edu_candidate:
        print(f"   Candidate: {', '.join(edu_candidate)}")
    print(f"   Required: {analysis['education_analysis']['required']}")
    print(f"   Match: {analysis['education_analysis']['match']}")
    
    print(f"\n📈 KEY METRICS:")
    for metric, value in analysis['key_metrics'].items():
        metric_name = metric.replace('_', ' ').title()
        if 'ratio' in metric or 'match' in metric or 'similarity' in metric:
            display_value = f"{value*100:.1f}%"
        else:
            display_value = f"{value:.3f}"
        print(f"   {metric_name}: {display_value}")
    
    if analysis['detailed_recommendations']:
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['detailed_recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


def save_report(score: float, analysis: Dict, cv_text: str, jd_text: str, 
                filename: Optional[str] = None):
    """
    Save detailed report to JSON file.
    
    Args:
        score: Predicted score
        analysis: Analysis dictionary
        cv_text: Original CV text
        jd_text: Original JD text
        filename: Output filename (optional)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fit_report_{timestamp}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'score': score,
        'analysis': analysis,
        'input': {
            'cv_preview': cv_text[:500] + "..." if len(cv_text) > 500 else cv_text,
            'jd_preview': jd_text[:500] + "..." if len(jd_text) > 500 else jd_text
        },
        'metadata': {
            'model_version': '2.0',
            'predictor': 'CandidateFitPredictor'
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved to: {filename}")
    return filename


def main():
    """Main execution function."""
    display_banner()
    
    # Initialize predictor
    print("[INFO] Initializing Candidate Fit Predictor...")
    predictor = CandidateFitPredictor()
    
    # Check for existing model
    import os
    model_file = "candidate_fit_model.pkl"
    
    if os.path.exists(model_file):
        print("[INFO] Loading pre-trained model...")
        predictor.load_model(model_file)
    else:
        print("[INFO] Training new model...")
        metrics = predictor.train(n_samples=200)
        predictor.save_model(model_file)
    
    print("\n" + "="*70)
    print("INPUT OPTIONS")
    print("="*70)
    
    use_sample = input("\nUse sample data for quick demo? (y/n): ").lower().strip()
    
    if use_sample == 'y':
        # Sample data
        cv_text = """
        Alexandra Chen
        Senior Data Scientist
        
        Summary:
        Experienced Data Scientist with 5 years in machine learning and AI.
        Strong background in Python, TensorFlow, and cloud technologies.
        
        Skills:
        Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL,
        AWS, Docker, Kubernetes, Pandas, NumPy, Scikit-learn, Statistics,
        Data Visualization, Tableau, Spark, Big Data
        
        Experience:
        5 years as Senior Data Scientist at TechInnovate Inc.
        Developed ML models for predictive analytics.
        
        Education:
        Master of Science in Computer Science - Stanford University
        Bachelor of Science in Statistics - UC Berkeley
        
        Certifications:
        AWS Certified Machine Learning - Specialty
        Google Cloud Professional Data Engineer
        """
        
        jd_text = """
        Senior Data Scientist - AI Research Team
        
        Job Description:
        We're looking for an experienced Senior Data Scientist to join our AI Research team.
        
        Requirements:
        - 4+ years experience in data science or machine learning roles
        - Strong proficiency in Python and ML libraries (TensorFlow, PyTorch)
        - Experience with cloud platforms (AWS, GCP, or Azure)
        - Solid understanding of statistics and data analysis
        - Master's degree in Computer Science, Statistics, or related field
        
        Responsibilities:
        - Develop and deploy machine learning models in production
        - Conduct research on new AI algorithms and techniques
        - Collaborate with engineering teams to implement solutions
        
        Required Skills:
        Python, Machine Learning, TensorFlow, SQL, AWS, Statistics,
        Data Analysis, Deep Learning, Cloud Computing
        
        Preferred Qualifications:
        - PhD in relevant field
        - Experience with Docker and Kubernetes
        - Published research in AI/ML conferences
        """
        
        print("\n[INFO] Using sample data...")
    
    else:
        print("\n📄 ENTER CV/RESUME TEXT")
        print("="*40)
        print("Paste your CV text below. Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done.")
        print("Include: Skills, Experience, Education sections\n")
        
        cv_lines = []
        try:
            while True:
                line = input()
                cv_lines.append(line)
        except EOFError:
            pass
        
        cv_text = "\n".join(cv_lines)
        
        print("\n📋 ENTER JOB DESCRIPTION")
        print("="*40)
        print("Paste the job description text below. Press Ctrl+D/Ctrl+Z when done.\n")
        
        jd_lines = []
        try:
            while True:
                line = input()
                jd_lines.append(line)
        except EOFError:
            pass
        
        jd_text = "\n".join(jd_lines)
    
    # Make prediction
    print("\n" + "="*70)
    print("ANALYZING COMPATIBILITY")
    print("="*70)
    
    try:
        score, analysis = predictor.predict(cv_text, jd_text)
        display_results(score, analysis)
        
        # Option to save report
        save_option = input("\n💾 Save detailed report? (y/n): ").lower().strip()
        if save_option == 'y':
            report_file = save_report(score, analysis, cv_text, jd_text)
            print(f"[SUCCESS] Report saved to {report_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        print("[INFO] Please check your inputs and try again.")
    
    print("\n" + "="*70)
    print("THANK YOU FOR USING CANDIDATE FIT PREDICTOR!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Program interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        logger.exception("Main execution failed")
        sys.exit(1)