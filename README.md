# Library Book Recommendation System Using Machine Learning

A Streamlit web application for the internship project. It includes:
- Home page and catalogue summary
- Book search by title, author, category, and keywords
- Category filtering
- Content-based book recommendations
- Project overview and module descriptions

## Run locally

1. Install Python 3.9 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Publish online with Streamlit Community Cloud

1. Create a GitHub account and a new repository.
2. Upload `app.py`, `books.csv`, and `requirements.txt`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose the repository, branch, and `app.py` as the main file.
5. Deploy the app. Streamlit will provide a public website URL.

## Machine Learning method

The demo combines book title, author, genre, description, and tags. Scikit-learn's
TF-IDF vectorizer converts this text into numerical features. Cosine similarity
compares the books and returns the most similar titles.

## Important project note

The included `books.csv` is a small sample dataset for demonstration. Replace it
with a verified dataset or your actual library catalogue before claiming that the
system uses real library records. The current recommendation score measures text
similarity; it is not a user rating or a validated prediction of preference.
