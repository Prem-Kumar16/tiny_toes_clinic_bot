# 👣👶🏻 Tiny Toes Clinical chatbot-cum-doctor appointment booking system

A simple Streamlit web chatbot app created using Google Gemma 2 2B Open source SLM and a 4500 page medical encyclopaedia called "The-Gale-Encyclopedia-of-Medicine-3rd-Edition" and PubMed opensource medical database as data source.
The framework used is **Retrieval-augmented generation (RAG)** 

### Highlights
1. Upto date medical data.
2. Retains upto 5 chat history to provide relevant and concurrent responses.
3. Intent and entity recognition for appointment booking.
4. Autofill the relevant details in appointment booking page using the entity recognized from user's chat.
5. User friendly interface.
6. Extremely lightweight mechanism.

   
![Screenshot (83)](https://github.com/user-attachments/assets/60213e90-c8f9-4bb5-8fc2-d85ef3c96a63)


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

3. Locally download the model (Used Llamacpp imatrix Quantizations of gemma-2-2b-it from bartowski - https://huggingface.co/bartowski/gemma-2-2b-it-GGUF The original model huggingface site is https://huggingface.co/google/gemma-2-2b-it). This is a gated model, so request permission for this model in https://huggingface.co/google/gemma-2-2b-it and follow the below steps to download the model locally.

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
   

![Screenshot (84)](https://github.com/user-attachments/assets/31fff359-f874-4299-8acf-d6b201d95f2b)


### Demo working video




https://github.com/user-attachments/assets/56472185-fe32-46ed-83e3-59dfd4284e93

