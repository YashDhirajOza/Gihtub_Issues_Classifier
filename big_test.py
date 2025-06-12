import joblib
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ModelTester:
    def __init__(self, model_path='large_github_issue_classifier.pkl'):
        """
        Initialize the model tester
        """
        self.model_path = model_path
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
        self.label_encoder = None
        self.priority_keywords = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_model(self):
        """
        Load the pre-trained model
        """
        try:
            print(f"📦 Loading model from {self.model_path}...")
            model_data = joblib.load(self.model_path)
            
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.priority_keywords = model_data.get('priority_keywords', {})
            
            print(f"✅ Model loaded successfully!")
            print(f"🏆 Best Model: {self.best_model_name}")
            print(f"🏷️  Classes: {list(self.label_encoder.classes_)}")
            print(f"📊 TF-IDF Features: {len(self.vectorizer.get_feature_names_out()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data (same as training)
        """
        if pd.isna(text) or text is None:
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
        
        # Remove stopwords (simplified for consistency with training)
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict_priority(self, title, body=""):
        """
        Predict priority for a new issue
        """
        if self.best_model is None:
            print("❌ Model not loaded. Please run load_model() first.")
            return None
        
        try:
            # Preprocess input (same way as training)
            processed_title = self.preprocess_text(title)
            processed_body = self.preprocess_text(body)
            
            # Combine features (title appears twice for weighting, same as training)
            combined_text = processed_title + ' ' + processed_title + ' ' + processed_body
            
            # Create TF-IDF features
            text_features = self.vectorizer.transform([combined_text])
            
            # For the Random Forest model trained on sparse matrices, use only text features
            features = text_features
            
            # Make prediction
            prediction = self.best_model.predict(features)[0]
            predicted_priority = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get probabilities
            try:
                probabilities = self.best_model.predict_proba(features)[0]
                prob_dict = {}
                for i, class_name in enumerate(self.label_encoder.classes_):
                    prob_dict[class_name] = probabilities[i]
            except Exception as e:
                print(f"Warning: Could not get probabilities: {e}")
                prob_dict = {predicted_priority: 1.0}
            
            return predicted_priority, prob_dict
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def test_with_examples(self):
        """
        Test the model with predefined examples
        """
        print("\n" + "="*70)
        print("🎯 TESTING SAVED MODEL WITH EXAMPLES")
        print("="*70)
        
        test_cases = [
            {
                'title': 'Application crashes when clicking save button',
                'body': 'The application crashes immediately when I click the save button. This is a critical bug that prevents users from saving their work. Error message: NullPointerException at line 245. This needs urgent attention as it affects all users.',
                'expected': 'HIGH'
            },
            {
                'title': 'Critical security vulnerability in authentication system',
                'body': 'Found a critical security flaw that allows unauthorized access. This needs immediate attention as it affects all users and could lead to data breaches. The vulnerability is in the login system.',
                'expected': 'HIGH'
            },
            {
                'title': 'Add support for dark theme',
                'body': 'Users have been requesting a dark theme option. This would improve user experience especially for users working in low-light environments. This is a feature request that would enhance usability.',
                'expected': 'MEDIUM'
            },
            {
                'title': 'Improve performance of search functionality',
                'body': 'The search feature is quite slow when dealing with large datasets. We should optimize the search algorithm to provide faster results. This is a performance improvement.',
                'expected': 'MEDIUM'
            },
            {
                'title': 'Fix typo in readme file',
                'body': 'There is a small typo in the README.md file. Line 23 says "installtion" instead of "installation". Minor documentation fix needed.',
                'expected': 'LOW'
            },
            {
                'title': 'Update documentation for new API endpoints',
                'body': 'The README file needs to be updated with information about the new API endpoints added in version 2.1. This is a documentation update that would help developers.',
                'expected': 'LOW'
            }
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}:")
            print(f"Title: {test_case['title']}")
            print(f"Body: {test_case['body'][:100]}...")
            print(f"Expected: {test_case['expected']}")
            
            result = self.predict_priority(test_case['title'], test_case['body'])
            
            if result:
                predicted_priority, probabilities = result
                print(f"🎯 Predicted: {predicted_priority}")
                
                # Check if prediction is correct
                if predicted_priority == test_case['expected']:
                    print("✅ CORRECT!")
                    correct_predictions += 1
                else:
                    print("❌ INCORRECT")
                
                print("📊 Probabilities:")
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                for priority, prob in sorted_probs:
                    print(f"   {priority}: {prob:.3f}")
            else:
                print("❌ Failed to make prediction")
        
        # Summary
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n" + "="*70)
        print(f"📊 TEST RESULTS SUMMARY")
        print(f"✅ Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"🎯 Accuracy: {accuracy:.1f}%")
        print("="*70)
    
    def interactive_test(self):
        """
        Interactive testing mode
        """
        print("\n" + "="*70)
        print("🎮 INTERACTIVE TESTING MODE")
        print("Enter issue details to get priority predictions")
        print("Type 'quit' to exit")
        print("="*70)
        
        while True:
            print("\n" + "-"*50)
            title = input("📝 Enter issue title: ").strip()
            
            if title.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not title:
                print("❌ Title cannot be empty!")
                continue
            
            body = input("📄 Enter issue body (optional): ").strip()
            
            print("\n🔄 Making prediction...")
            result = self.predict_priority(title, body)
            
            if result:
                predicted_priority, probabilities = result
                print(f"\n🎯 Predicted Priority: {predicted_priority}")
                print("📊 Confidence Scores:")
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                for priority, prob in sorted_probs:
                    bar = "█" * int(prob * 20)  # Simple progress bar
                    print(f"   {priority}: {prob:.3f} {bar}")
            else:
                print("❌ Failed to make prediction")
    
    def batch_test_from_csv(self, csv_path, title_col='issue_title', body_col='body', n_samples=10):
        """
        Test on a batch of issues from a CSV file
        """
        try:
            print(f"\n📂 Loading test data from {csv_path}...")
            df = pd.read_csv(csv_path, nrows=n_samples)
            
            if title_col not in df.columns:
                print(f"❌ Column '{title_col}' not found in CSV")
                return
            
            body_col_exists = body_col in df.columns
            
            print(f"🔍 Testing on {len(df)} samples...")
            
            for i, row in df.iterrows():
                title = row[title_col]
                body = row[body_col] if body_col_exists else ""
                
                print(f"\n--- Sample {i+1} ---")
                print(f"Title: {title}")
                if body:
                    print(f"Body: {str(body)[:100]}...")
                
                result = self.predict_priority(title, body)
                if result:
                    predicted_priority, probabilities = result
                    print(f"🎯 Predicted: {predicted_priority}")
                    print(f"📊 Confidence: {max(probabilities.values()):.3f}")
                else:
                    print("❌ Failed to predict")
                    
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")

def main():
    """
    Main function to run the model tester
    """
    print("🚀 GitHub Issue Priority Classifier - Model Tester")
    print("="*70)
    
    # Initialize tester
    tester = ModelTester('large_github_issue_classifier.pkl')
    
    # Load model
    if not tester.load_model():
        print("❌ Failed to load model. Make sure the model file exists.")
        return
    
    while True:
        print("\n🎯 What would you like to do?")
        print("1. Test with predefined examples")
        print("2. Interactive testing")
        print("3. Batch test from CSV file")
        print("4. Single prediction")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            tester.test_with_examples()
        
        elif choice == '2':
            tester.interactive_test()
        
        elif choice == '3':
            csv_path = input("Enter CSV file path: ").strip()
            n_samples = input("Number of samples to test (default 10): ").strip()
            n_samples = int(n_samples) if n_samples.isdigit() else 10
            tester.batch_test_from_csv(csv_path, n_samples=n_samples)
        
        elif choice == '4':
            title = input("Enter issue title: ").strip()
            body = input("Enter issue body (optional): ").strip()
            
            if title:
                result = tester.predict_priority(title, body)
                if result:
                    predicted_priority, probabilities = result
                    print(f"\n🎯 Predicted Priority: {predicted_priority}")
                    print("📊 Probabilities:")
                    for priority, prob in probabilities.items():
                        print(f"   {priority}: {prob:.3f}")
            else:
                print("❌ Title cannot be empty!")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()