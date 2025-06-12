import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import warnings
import gc
import os
import time
from typing import Iterator, Tuple
import psutil
import multiprocessing
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class LargeDatasetGitHubIssueClassifier:
    def __init__(self, csv_path='github_issues.csv', chunk_size=10000):
        """
        Initialize the GitHub Issue Priority Classifier for large datasets
        
        Args:
            csv_path: Path to the CSV file
            chunk_size: Number of rows to process at once
        """
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.df = None
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.total_rows = 0
        
        # Priority keywords for automatic labeling
        self.priority_keywords = {
            'HIGH': {
                'primary': ['crash', 'critical', 'urgent', 'security', 'vulnerability', 'blocker', 
                           'data loss', 'system down', 'production', 'fatal', 'severe', 'emergency'],
                'secondary': ['bug', 'error', 'broken', 'fails', 'exception', 'null pointer', 
                             'memory leak', 'timeout', 'corrupt', 'freeze', 'hang']
            },
            'MEDIUM': {
                'primary': ['improvement', 'enhancement', 'feature request', 'performance', 
                           'optimization', 'usability', 'ui', 'ux', 'refactor'],
                'secondary': ['slow', 'missing', 'add', 'support', 'implement', 'update', 
                             'modify', 'change', 'request', 'proposal']
            },
            'LOW': {
                'primary': ['documentation', 'typo', 'cosmetic', 'style', 'formatting', 
                           'comment', 'readme', 'help', 'question', 'docs'],
                'secondary': ['suggestion', 'idea', 'minor', 'cleanup', 'polish', 
                             'nice to have', 'wishlist', 'discussion']
            }
        }
    
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def estimate_dataset_size(self):
        """Estimate dataset size and provide recommendations"""
        print("📊 Analyzing dataset size...")
        
        # Get file size
        file_size = os.path.getsize(self.csv_path) / (1024**3)  # GB
        print(f"📁 File size: {file_size:.2f} GB")
        
        # Estimate number of rows by reading a small sample
        sample_df = pd.read_csv(self.csv_path, nrows=1000)
        avg_row_size = len(sample_df.to_csv()) / len(sample_df)
        estimated_rows = int(file_size * 1024**3 / avg_row_size)
        
        print(f"📏 Estimated rows: {estimated_rows:,}")
        print(f"📋 Columns: {list(sample_df.columns)}")
        
        # Memory recommendations
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        print(f"💾 Available RAM: {available_memory:.2f} GB")
        
        if file_size > available_memory * 0.5:
            print("⚠️  Large dataset detected! Using chunked processing...")
            recommended_chunk_size = max(1000, int(available_memory * 1024**2 / avg_row_size / 10))
            self.chunk_size = min(self.chunk_size, recommended_chunk_size)
            print(f"🔧 Recommended chunk size: {self.chunk_size:,}")
        
        return estimated_rows, sample_df.columns.tolist()
    
    def preprocess_text_batch(self, texts):
        """
        Batch preprocess text data for efficiency
        """
        processed_texts = []
        
        for text in texts:
            if pd.isna(text):
                processed_texts.append("")
                continue
                
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
            
            # Remove stopwords and stem (simplified for speed)
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            processed_texts.append(' '.join(words))
        
        return processed_texts
    
    def auto_label_priority_batch(self, titles, bodies):
        """
        Batch auto-label priorities for efficiency
        """
        priorities = []
        
        for title, body in zip(titles, bodies):
            # Combine title and body
            combined_text = f"{str(title).lower()} {str(body).lower()}"
            
            score = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            # Score based on keywords
            for priority, keywords in self.priority_keywords.items():
                for keyword in keywords['primary']:
                    if keyword in combined_text:
                        score[priority] += 3
                
                for keyword in keywords['secondary']:
                    if keyword in combined_text:
                        score[priority] += 1
            
            # Assign priority based on highest score
            if score['HIGH'] >= 3:
                priorities.append('HIGH')
            elif score['MEDIUM'] >= 2:
                priorities.append('MEDIUM')
            elif score['LOW'] >= 1:
                priorities.append('LOW')
            else:
                # Default assignment based on content length
                if len(combined_text.split()) > 100:
                    priorities.append('MEDIUM')
                else:
                    priorities.append('LOW')
        
        return priorities
    
    def process_data_in_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Process data in chunks to handle large datasets
        """
        print(f"🔄 Processing data in chunks of {self.chunk_size:,} rows...")
        
        chunk_count = 0
        total_processed = 0
        
        try:
            for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
                chunk_count += 1
                total_processed += len(chunk)
                
                print(f"📦 Processing chunk {chunk_count} ({total_processed:,} rows processed)")
                print(f"💾 Memory usage: {self.get_memory_usage():.1f} MB")
                
                # Auto-label priorities
                priorities = self.auto_label_priority_batch(
                    chunk['issue_title'].fillna(''),
                    chunk['body'].fillna('')
                )
                chunk['priority'] = priorities
                
                # Preprocess text
                chunk['processed_title'] = self.preprocess_text_batch(chunk['issue_title'].fillna(''))
                chunk['processed_body'] = self.preprocess_text_batch(chunk['body'].fillna(''))
                
                # Combine features
                chunk['combined_features'] = (
                    chunk['processed_title'] + ' ' + 
                    chunk['processed_title'] + ' ' +  # Title appears twice for weighting
                    chunk['processed_body']
                )
                
                # Additional features
                chunk['title_length'] = chunk['issue_title'].str.len()
                chunk['body_length'] = chunk['body'].str.len()
                chunk['word_count'] = chunk['combined_features'].str.split().str.len()
                
                yield chunk
                
                # Force garbage collection
                gc.collect()
                
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_count}: {e}")
            raise
    
    def create_sample_dataset(self, sample_size=50000, output_path='github_issues_sample.csv'):
        """
        Create a representative sample of the large dataset
        """
        print(f"🎯 Creating sample dataset of {sample_size:,} rows...")
        
        total_chunks = 0
        sampled_chunks = []
        
        # Calculate sampling ratio
        estimated_rows, _ = self.estimate_dataset_size()
        sampling_ratio = min(1.0, sample_size / estimated_rows)
        
        print(f"📊 Sampling ratio: {sampling_ratio:.4f}")
        
        for chunk in self.process_data_in_chunks():
            total_chunks += 1
            
            # Sample from this chunk
            chunk_sample_size = int(len(chunk) * sampling_ratio)
            if chunk_sample_size > 0:
                chunk_sample = chunk.sample(n=min(chunk_sample_size, len(chunk)), random_state=42)
                sampled_chunks.append(chunk_sample)
            
            # Check if we have enough samples
            total_sampled = sum(len(chunk) for chunk in sampled_chunks)
            if total_sampled >= sample_size:
                break
        
        # Combine all sampled chunks
        if sampled_chunks:
            sample_df = pd.concat(sampled_chunks, ignore_index=True)
            
            # Ensure we don't exceed the desired sample size
            if len(sample_df) > sample_size:
                sample_df = sample_df.sample(n=sample_size, random_state=42)
            
            # Save sample
            sample_df.to_csv(output_path, index=False)
            
            print(f"✅ Sample dataset created: {output_path}")
            print(f"📊 Sample size: {len(sample_df):,} rows")
            
            # Show priority distribution
            priority_counts = sample_df['priority'].value_counts()
            print("\n📊 Priority Distribution in Sample:")
            for priority, count in priority_counts.items():
                percentage = (count / len(sample_df)) * 100
                print(f"  {priority}: {count:,} ({percentage:.1f}%)")
            
            return sample_df
        else:
            print("❌ No samples could be created")
            return None
    
    def train_on_sample(self, sample_df=None, sample_path='github_issues_sample.csv'):
        """
        Train models on a sample dataset
        """
        if sample_df is None:
            try:
                sample_df = pd.read_csv(sample_path)
                print(f"✅ Loaded sample dataset: {sample_path}")
            except FileNotFoundError:
                print(f"❌ Sample file not found: {sample_path}")
                print("Creating sample dataset...")
                sample_df = self.create_sample_dataset(output_path=sample_path)
                if sample_df is None:
                    return False
        
        print(f"🤖 Training models on sample dataset ({len(sample_df):,} rows)...")
        
        # Prepare data
        X_text = sample_df['combined_features']
        X_numeric = sample_df[['title_length', 'body_length', 'word_count']].fillna(0)
        y = sample_df['priority']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
            X_text, X_numeric, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # TF-IDF Vectorization with memory optimization
        print("🔤 Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased for better performance
            ngram_range=(1, 2),
            min_df=5,  # Increased to reduce features
            max_df=0.7,  # Reduced to exclude very common terms
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        X_test_tfidf = self.vectorizer.transform(X_test_text)
        
        print(f"📊 TF-IDF feature matrix shape: {X_train_tfidf.shape}")
        
        # For large datasets, use models that handle sparse matrices well
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'SGD Classifier': SGDClassifier(random_state=42, loss='log_loss', max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            start_time = time.time()
            
            try:
                # Train model
                if name in ['Naive Bayes', 'SGD Classifier', 'Logistic Regression']:
                    # These models work well with sparse matrices
                    model.fit(X_train_tfidf, y_train)
                    y_pred = model.predict(X_test_tfidf)
                else:
                    # Convert to dense for tree-based models (with memory consideration)
                    if X_train_tfidf.shape[1] > 5000:
                        print(f"    Note: Using sparse matrix for {name} due to size")
                        model.fit(X_train_tfidf, y_train)
                        y_pred = model.predict(X_test_tfidf)
                    else:
                        X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_num.values])
                        X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_num.values])
                        model.fit(X_train_combined, y_train)
                        y_pred = model.predict(X_test_combined)
                
                # Evaluate
                accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'predictions': y_pred
                }
                
                self.models[name] = model
                
                print(f"    Accuracy: {accuracy:.3f} (trained in {training_time:.1f}s)")
                
            except Exception as e:
                print(f"    ❌ Error training {name}: {e}")
                continue
        
        if not results:
            print("❌ No models were successfully trained")
            return False
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")
        
        # Detailed evaluation
        print(f"\n📊 Detailed Evaluation - {best_model_name}:")
        y_pred_best = results[best_model_name]['predictions']
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_best, 
                                  target_names=self.label_encoder.classes_))
        
        # Show confusion matrix
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
        
        return True
    
    def predict_priority(self, title, body):
        """
        Predict priority for a new issue
        """
        if not hasattr(self, 'best_model'):
            print("❌ Model not trained yet. Please run training first.")
            return None
        
        # Preprocess input
        processed_title = self.preprocess_text_batch([title])[0]
        processed_body = self.preprocess_text_batch([body])[0]
        combined_text = processed_title + ' ' + processed_title + ' ' + processed_body
        
        # Create features
        text_features = self.vectorizer.transform([combined_text])
        
        if self.best_model_name in ['Naive Bayes', 'SGD Classifier', 'Logistic Regression']:
            features = text_features
        else:
            try:
                numeric_features = np.array([[len(title), len(body), len(combined_text.split())]])
                features = np.hstack([text_features.toarray(), numeric_features])
            except:
                features = text_features
        
        # Predict
        prediction = self.best_model.predict(features)[0]
        
        try:
            probability = self.best_model.predict_proba(features)[0]
            prob_dict = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[class_name] = probability[i]
        except:
            prob_dict = {self.label_encoder.inverse_transform([prediction])[0]: 1.0}
        
        predicted_priority = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_priority, prob_dict
    
    def save_model(self, filename='large_github_issue_classifier.pkl'):
        """
        Save the trained model
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'priority_keywords': self.priority_keywords,
            'chunk_size': self.chunk_size
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Model saved as {filename}")
    
    def load_model(self, filename='large_github_issue_classifier.pkl'):
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
            self.chunk_size = model_data.get('chunk_size', 10000)
            print(f"✅ Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def run_full_pipeline(self, use_sample=True, sample_size=50000):
        """
        Run the complete training pipeline for large datasets
        """
        print("🚀 Starting Large Dataset GitHub Issues Priority Classification Pipeline")
        print("=" * 70)
        
        # Step 1: Analyze dataset
        estimated_rows, columns = self.estimate_dataset_size()
        
        if use_sample:
            # Step 2: Create and train on sample
            print(f"\n📊 Creating sample dataset ({sample_size:,} rows)...")
            sample_df = self.create_sample_dataset(sample_size=sample_size)
            
            if sample_df is not None:
                # Step 3: Train on sample
                if self.train_on_sample(sample_df):
                    # Step 4: Save model
                    self.save_model()
                    print("\n" + "=" * 70)
                    print("🎉 Pipeline completed successfully!")
                    print(f"✅ Best model: {self.best_model_name}")
                    print("✅ Model saved for future use")
                    return True
        else:
            print("⚠️  Training on full dataset not recommended for 2.3GB file")
            print("💡 Consider using sample-based training instead")
        
        return False

def demo_large_classifier():
    """
    Demonstrate the large dataset classifier
    """
    print("\n" + "=" * 70)
    print("🎯 DEMO: Testing the large dataset classifier")
    print("=" * 70)
    
    # Initialize classifier
    classifier = LargeDatasetGitHubIssueClassifier(csv_path='github_issues.csv')
    
    # Try to load existing model
    if not classifier.load_model('large_github_issue_classifier.pkl'):
        print("No existing model found. Training new model...")
        if not classifier.run_full_pipeline(use_sample=True, sample_size=50000):
            print("❌ Failed to train model")
            return
    
    # Test cases
    test_cases = [
        {
            'title': 'Critical security vulnerability in authentication system',
            'body': 'Found a critical security flaw that allows unauthorized access. This needs immediate attention as it affects all users and could lead to data breaches.'
        },
        {
            'title': 'Add support for dark theme',
            'body': 'Users have been requesting a dark theme option. This would improve user experience especially for users working in low-light environments.'
        },
        {
            'title': 'Update documentation for new API endpoints',
            'body': 'The README file needs to be updated with information about the new API endpoints added in version 2.1. This is a documentation update.'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"Title: {test_case['title']}")
        print(f"Body: {test_case['body'][:100]}...")
        
        result = classifier.predict_priority(test_case['title'], test_case['body'])
        if result:
            predicted_priority, probabilities = result
            print(f"🎯 Predicted Priority: {predicted_priority}")
            print("📊 Probabilities:")
            for priority, prob in probabilities.items():
                print(f"   {priority}: {prob:.3f}")

if __name__ == "__main__":
    # For large dataset (2.3GB), use sample-based training
    classifier = LargeDatasetGitHubIssueClassifier(csv_path='github_issues.csv', chunk_size=5000)
    
    # Run the pipeline with sample-based training
    classifier.run_full_pipeline(use_sample=True, sample_size=100000)  # 100k sample
    
    # Run demo
    demo_large_classifier()