import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class GitHubIssueClassifier:
    def __init__(self, csv_path='github_issues_sample.csv'):
        """
        Initialize the GitHub Issue Priority Classifier
        """
        self.csv_path = csv_path
        self.df = None
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Priority keywords for automatic labeling
        self.priority_keywords = {
            'HIGH': {
                'primary': ['crash', 'critical', 'urgent', 'security', 'vulnerability', 'blocker', 
                           'data loss', 'system down', 'production', 'fatal', 'severe'],
                'secondary': ['bug', 'error', 'broken', 'fails', 'exception', 'null pointer', 
                             'memory leak', 'timeout', 'corrupt']
            },
            'MEDIUM': {
                'primary': ['improvement', 'enhancement', 'feature request', 'performance', 
                           'optimization', 'usability', 'ui', 'ux'],
                'secondary': ['slow', 'missing', 'add', 'support', 'implement', 'update', 
                             'modify', 'change']
            },
            'LOW': {
                'primary': ['documentation', 'typo', 'cosmetic', 'style', 'formatting', 
                           'comment', 'readme', 'help', 'question'],
                'secondary': ['suggestion', 'idea', 'minor', 'cleanup', 'refactor', 
                             'polish', 'nice to have']
            }
        }
    
    def load_data(self):
        """Load and display basic information about the dataset"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove code blocks (common in GitHub issues)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and stem
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def auto_label_priority(self):
        """
        Automatically assign priority labels based on keywords
        """
        print("🏷️  Auto-labeling issues based on keywords...")
        
        # Combine title and body for analysis
        self.df['combined_text'] = (self.df['issue_title'].fillna('') + ' ' + 
                                   self.df['body'].fillna('')).str.lower()
        
        priorities = []
        
        for text in self.df['combined_text']:
            score = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            # Score based on keywords
            for priority, keywords in self.priority_keywords.items():
                for keyword in keywords['primary']:
                    if keyword in text:
                        score[priority] += 3
                
                for keyword in keywords['secondary']:
                    if keyword in text:
                        score[priority] += 1
            
            # Assign priority based on highest score
            if score['HIGH'] >= 3:
                priorities.append('HIGH')
            elif score['MEDIUM'] >= 2:
                priorities.append('MEDIUM')
            elif score['LOW'] >= 1:
                priorities.append('LOW')
            else:
                # Default assignment based on content length and complexity
                if len(text.split()) > 100:
                    priorities.append('MEDIUM')
                else:
                    priorities.append('LOW')
        
        self.df['priority'] = priorities
        
        # Display distribution
        priority_counts = self.df['priority'].value_counts()
        print("\n📊 Priority Distribution:")
        for priority, count in priority_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {priority}: {count} ({percentage:.1f}%)")
        
        return self.df
    
    def prepare_features(self):
        """
        Prepare features for machine learning
        """
        print("🔧 Preparing features...")
        
        # Preprocess text
        self.df['processed_title'] = self.df['issue_title'].apply(self.preprocess_text)
        self.df['processed_body'] = self.df['body'].apply(self.preprocess_text)
        
        # Combine title and body (title is more important, so we weight it)
        self.df['combined_features'] = (self.df['processed_title'] + ' ' + 
                                       self.df['processed_title'] + ' ' +  # Title appears twice for weighting
                                       self.df['processed_body'])
        
        # Additional features
        self.df['title_length'] = self.df['issue_title'].str.len()
        self.df['body_length'] = self.df['body'].str.len()
        self.df['word_count'] = self.df['combined_features'].str.split().str.len()
        
        print("✅ Features prepared successfully!")
        return self.df
    
    def train_models(self):
        """
        Train multiple ML models and compare their performance
        """
        print("🤖 Training machine learning models...")
        
        # Prepare data
        X_text = self.df['combined_features']
        X_numeric = self.df[['title_length', 'body_length', 'word_count']]
        y = self.df['priority']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
            X_text, X_numeric, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        X_test_tfidf = self.vectorizer.transform(X_test_text)
        
        # Combine text and numeric features
        X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_num.values])
        X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_num.values])
        
        # Define models
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            
            # Train model
            if name == 'Naive Bayes':
                # Naive Bayes needs non-negative features
                model.fit(X_train_tfidf, y_train)
                y_pred = model.predict(X_test_tfidf)
            else:
                model.fit(X_train_combined, y_train)
                y_pred = model.predict(X_test_combined)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            self.models[name] = model
            
            print(f"    Accuracy: {accuracy:.3f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")
        
        # Detailed evaluation of best model
        print(f"\n📊 Detailed Evaluation - {best_model_name}:")
        y_pred_best = results[best_model_name]['predictions']
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_best, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
        
        return results
    
    def plot_feature_importance(self):
        """
        Plot feature importance for tree-based models
        """
        if not hasattr(self.best_model, 'feature_importances_'):
            return
        
        # Get feature names
        feature_names = list(self.vectorizer.get_feature_names_out()) + ['title_length', 'body_length', 'word_count']
        
        # Get top 20 most important features
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {self.best_model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def predict_priority(self, title, body):
        """
        Predict priority for a new issue
        """
        if not hasattr(self, 'best_model'):
            print("❌ Model not trained yet. Please run train_models() first.")
            return None
        
        # Preprocess input
        processed_title = self.preprocess_text(title)
        processed_body = self.preprocess_text(body)
        combined_text = processed_title + ' ' + processed_title + ' ' + processed_body
        
        # Create features
        text_features = self.vectorizer.transform([combined_text])
        numeric_features = np.array([[len(title), len(body), len(combined_text.split())]])
        
        if self.best_model_name == 'Naive Bayes':
            features = text_features
        else:
            features = np.hstack([text_features.toarray(), numeric_features])
        
        # Predict
        prediction = self.best_model.predict(features)[0]
        probability = self.best_model.predict_proba(features)[0]
        
        predicted_priority = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities for each class
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probability[i]
        
        return predicted_priority, prob_dict
    
    def save_model(self, filename='github_issue_classifier.pkl'):
        """
        Save the trained model
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'priority_keywords': self.priority_keywords
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Model saved as {filename}")
    
    def load_model(self, filename='github_issue_classifier.pkl'):
        """
        Load a pre-trained model
        """
        try:
            model_data = joblib.load(filename)
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.priority_keywords = model_data['priority_keywords']
            print(f"✅ Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def run_full_pipeline(self):
        """
        Run the complete training pipeline
        """
        print("🚀 Starting GitHub Issues Priority Classification Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Auto-label priorities
        self.auto_label_priority()
        
        # Step 3: Prepare features
        self.prepare_features()
        
        # Step 4: Train models
        results = self.train_models()
        
        # Step 5: Save model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("🎉 Pipeline completed successfully!")
        print(f"✅ Best model: {self.best_model_name}")
        print("✅ Model saved for future use")
        
        return True

# Demo usage
def demo_classifier():
    """
    Demonstrate the classifier with example predictions
    """
    print("\n" + "=" * 60)
    print("🎯 DEMO: Testing the classifier")
    print("=" * 60)
    
    # Initialize classifier
    classifier = GitHubIssueClassifier()
    
    # Try to load existing model
    if not classifier.load_model():
        print("No existing model found. Training new model...")
        if not classifier.run_full_pipeline():
            return
    
    # Test cases
    test_cases = [
        {
            'title': 'Application crashes when clicking save button',
            'body': 'The application crashes immediately when I click the save button. This is a critical bug that prevents users from saving their work. Error message: NullPointerException at line 245.'
        },
        {
            'title': 'Add dark mode support',
            'body': 'It would be nice to have a dark mode option for better user experience during night time usage. This is a feature request that could improve usability.'
        },
        {
            'title': 'Fix typo in readme file',
            'body': 'There is a small typo in the README.md file. Line 23 says "installtion" instead of "installation". Minor documentation fix needed.'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"Title: {test_case['title']}")
        print(f"Body: {test_case['body'][:100]}...")
        
        predicted_priority, probabilities = classifier.predict_priority(
            test_case['title'], test_case['body']
        )
        
        print(f"🎯 Predicted Priority: {predicted_priority}")
        print("📊 Probabilities:")
        for priority, prob in probabilities.items():
            print(f"   {priority}: {prob:.3f}")

if __name__ == "__main__":
    # Run the full pipeline
    classifier = GitHubIssueClassifier()
    classifier.run_full_pipeline()
    
    # Run demo
    demo_classifier()