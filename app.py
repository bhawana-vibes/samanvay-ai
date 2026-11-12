import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Page setup
st.set_page_config(page_title="Samanvay AI", page_icon="🤍", layout="centered")

st.title("🤍 Samanvay (समन्वय) – AI Empathy Companion")
st.subheader("Apne mann ki baat likhein, aur rishton mein apnapan badhayein")
st.write("---")

# Dataset load karna
df = pd.read_csv("data.csv")

# NLP Model Setup (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["sample_text"])

# User Input (Text Box)
user_input = st.text_area(
    "Abhi aap kaisa mehsoos kar rahe hain? (Hinglish ya English mein 1-2 line likhein):",
    placeholder="Jaise: Aaj kaam ki wajah se bohot thakaan aur chidchidapan ho raha hai...",
)

if st.button("Samanvay AI Sujhav Dekhein ✨"):
    if user_input.strip() == "":
        st.warning("Kripya pehle apne mann ki 1-2 line likhein.")
    else:
        # User ke text ko vector mein badalna aur best match nikaalna
        user_vec = vectorizer.transform([user_input.lower()])
        similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
        best_match_idx = similarity_scores.argmax()
        confidence = similarity_scores[0][best_match_idx]

        result = df.iloc[best_match_idx]

        # Agar text bilkul hi alag ho toh fallback message
        if confidence < 0.1:
            st.info(
                "Aapke shabd thode naye hain, par sabse pehle ek gehri saans lein aur bina phone ke 5 minute apno ke paas baithein. 🤍"
            )
        else:
            st.success(
                f"**Pehchana Gaya Mood:** {result['mood']} (Intensity: {result['intensity']})"
            )

            st.markdown("### 🌿 Aaj ka Offline Kadam (1-Minute Rule):")
            st.info(result["offline_task"])

            st.markdown("### 💬 Baat Shuru Karne ka Tareeka:")
            st.warning(f'"{result["conversation_starter"]}"')

        st.caption(
            "🔒 Zero-Storage Guarantee: Aapka likha hua text kahin bhi save nahi kiya gaya hai."
        )