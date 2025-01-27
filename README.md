# 💬 Tiny Toes Clinical chatbot-cum-doctor appointment booking system

A simple Streamlit web chatbot app created using Google Gemma 2 2B Open source SLM and a 4500 page medical encyclopaedia called "The-Gale-Encyclopedia-of-Medicine-3rd-Edition" and PubMed opensource medical database as data source.
The framework used is **Retrieval-augmented generation (RAG)** 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### Highlights
1. Upto date medical data.
2. Retains upto 5 chat history to provide relevant and concurrent responses.
3. Intent and entity recognition for appointment booking.
4. User friendly interface.
5. Extremely lightweight mechanism with BERTScore F1 ~0.86

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Getting the medical encyclopaedia as a pdf file

   ```
   $ mkdir data && cd data
   $ wget https://staibabussalamsula.ac.id/wp-content/uploads/2024/06/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf
   ```

3. Locally download the model. This is a gated model, so request permission for this model in https://huggingface.co/google/gemma-2-2b-it and follow the below steps to download the model locally

   ```
   $ cd ..
   $ mkdir model && cd model
   $ wget https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q6_K.gguf
   ```

4. Create FAISS Vector database

   ```
   $ cd ..
   $ mkdir vectorstore && cd vectorstore
   $ mkdir db_faiss && cd db_faiss
   $ cd ../..
   $ python3 ingest.py
   ```

5. Create local Doctor database for appointment booking

   ```
   $ sudo apt install sqlite3
   $ python3 create_db.py
   ```

6. Run the streamlit app locally

   ```
   $ streamlit run streamlit_app.py
   ```
   
