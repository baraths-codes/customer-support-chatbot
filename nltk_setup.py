# nltk_setup.py
import os
import nltk

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4',download_dir=nltk_data_path)
