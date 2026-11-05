import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(
    page_title="Library Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

DATA_FILE = Path(__file__).parent / "books.csv"

@st.cache_data
def load_books():
    return pd.read_csv(DATA_FILE).fillna("")

@st.cache_resource
def build_model(data):
    # Combine descriptive book fields so similar books can be identified.
    text_features = (
        data["title"].astype(str) + " " +
        data["author"].astype(str) + " " +
        data["genre"].astype(str) + " " +
        data["description"].astype(str) + " " +
        data["tags"].astype(str)
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(text_features)
    similarity = cosine_similarity(matrix, matrix)
    return similarity

books = load_books()
similarity_matrix = build_model(books)

def recommend_books(book_index, count=5, genre_filter="All"):
    scores = list(enumerate(similarity_matrix[book_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    results = []
    for idx, score in scores:
        if idx == book_index:
            continue
        row = books.iloc[idx]
        if genre_filter != "All" and row["genre"] != genre_filter:
            continue
        results.append((idx, float(score)))
        if len(results) >= count:
            break
    return results

st.title("📚 Library Book Recommendation System")
st.caption("A Machine Learning internship project — discover books based on your interests.")

with st.sidebar:
    st.header("🔎 Find your next book")
    page = st.radio("Navigate", ["Home", "Search Books", "Get Recommendations", "About Project"])
    st.divider()
    st.caption("Developed as an internship project")
    st.caption("Machine Learning • Python • Streamlit")

if page == "Home":
    st.subheader("Welcome to your smart library")
    st.write(
        "Search the sample library, explore book categories, or choose a book to receive "
        "personalized recommendations based on content similarity."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Books in catalogue", len(books))
    c2.metric("Book categories", books["genre"].nunique())
    c3.metric("Recommendation method", "TF-IDF")
    st.subheader("Popular books in this demo")
    cols = st.columns(3)
    for i, (_, row) in enumerate(books.head(6).iterrows()):
        with cols[i % 3]:
            st.markdown(f"**{row['title']}**")
            st.write(f"By {row['author']}")
            st.caption(f"{row['genre']} • ⭐ {row['rating']}")
            st.write(row["description"])

elif page == "Search Books":
    st.subheader("Search the book catalogue")
    query = st.text_input("Enter a title, author, genre, or keyword")
    genres = ["All"] + sorted(books["genre"].dropna().unique().tolist())
    selected_genre = st.selectbox("Filter by category", genres)
    filtered = books.copy()
    if selected_genre != "All":
        filtered = filtered[filtered["genre"] == selected_genre]
    if query.strip():
        q = query.strip().lower()
        mask = (
            filtered["title"].str.lower().str.contains(q, na=False) |
            filtered["author"].str.lower().str.contains(q, na=False) |
            filtered["genre"].str.lower().str.contains(q, na=False) |
            filtered["description"].str.lower().str.contains(q, na=False) |
            filtered["tags"].str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]
    st.write(f"**{len(filtered)} book(s) found**")
    if filtered.empty:
        st.info("No matching books found. Try another keyword or category.")
    else:
        for _, row in filtered.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row['title']}")
                st.write(f"**Author:** {row['author']}  |  **Category:** {row['genre']}  |  **Rating:** ⭐ {row['rating']}")
                st.write(row["description"])

elif page == "Get Recommendations":
    st.subheader("Get personalized book recommendations")
    st.write("Choose a book you like. The model compares its content with other books in the catalogue.")
    selected_title = st.selectbox("Choose a book", books["title"].tolist())
    genres = ["All"] + sorted(books["genre"].dropna().unique().tolist())
    genre_filter = st.selectbox("Optional: show recommendations from category", genres)
    number = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    if st.button("Recommend Books", type="primary"):
        selected_index = books.index[books["title"] == selected_title][0]
        selected = books.iloc[selected_index]
        st.success(f"Recommendations based on: {selected['title']}")
        st.markdown(f"**Selected book:** {selected['title']} — {selected['author']}")
        recs = recommend_books(selected_index, number, genre_filter)
        if not recs:
            st.warning("No similar books match this category. Choose 'All' or another category.")
        else:
            for rank, (idx, score) in enumerate(recs, start=1):
                row = books.iloc[idx]
                with st.container(border=True):
                    st.markdown(f"### {rank}. {row['title']}")
                    st.write(f"**Author:** {row['author']}  |  **Category:** {row['genre']}  |  **Rating:** ⭐ {row['rating']}")
                    st.write(row["description"])
                    st.caption(f"Content similarity score: {score:.2f} (higher means more similar in this demo)")

elif page == "About Project":
    st.subheader("Project overview")
    st.markdown("""
**Project title:** Library Book Recommendation System Using Machine Learning

**Objective:** Help users discover books that match their interests and reduce the time spent searching through a large library catalogue.

**Main modules**
1. **User Module:** Lets users search for books and choose a book they like.
2. **Book Data Module:** Stores book title, author, category, genre, rating, description, and tags.
3. **Data Preprocessing Module:** Reads the catalogue and prepares text fields for analysis.
4. **Recommendation Module:** Uses TF-IDF text features and cosine similarity to find books with similar content.
5. **Recommendation Display Module:** Displays recommended books and similarity scores.
6. **Evaluation Module:** Can be extended with user feedback and precision/recall evaluation.

**Technology used:** Python, Streamlit, Pandas, Scikit-learn.

**Note:** This downloadable demo uses a small sample catalogue. Replace `books.csv` with a verified library dataset for your final internship demonstration. The similarity score is a content-similarity measure, not a guarantee that a user will like a book.
""")

st.divider()
st.caption("Library Book Recommendation System • Internship Project")
