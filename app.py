import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Load the cleaned dataset
data = pd.read_csv("Dataset/cleaned_eng_subtitle.csv")

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Initialize ChromaDB Client and Set Up Embedding Function
chroma_client = PersistentClient(path="vectordb")
model_name = "all-mpnet-base-v2"
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# Retrieve the ChromaDB collection
collection_name = "eng_subtitles_collection"
try:
    collection = chroma_client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
except ValueError:
    st.error(f"Collection '{collection_name}' does not exist. Please add documents to the collection first.")
    st.stop()

# Define function to retrieve similar subtitles
def retrieve_similar_subtitles(user_query, top_n=10):
    # Preprocess the query
    preprocessed_query = user_query.lower()

    # Create query embedding
    query_embedding = sentence_model.encode([preprocessed_query])

    # Retrieve similar subtitles using ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_n,
        include=['metadatas', 'distances']
    )

    # Store results in a DataFrame
    similarity_scores = results['distances'][0]
    subtitles = pd.DataFrame(results['metadatas'][0], columns=['Subtitle_Id', 'Subtitle_Name'])
    subtitles['Similarity_Score'] = similarity_scores

    return subtitles

# Streamlit UI
st.title("Enhancing Search Engine Relevance for Video Subtitles 🎬")

# User input
user_input = st.text_input("Enter a movie or TV series name:", "")

# Button to trigger search
search_button = st.button("Search")

# Retrieve and display similar subtitles when the button is clicked
if search_button and user_input:
    similar_subtitles = retrieve_similar_subtitles(user_input)
    st.subheader("Top Ten Similar Movies:")
    st.write(similar_subtitles['Subtitle_Name'])
