import streamlit as st
import sqlite3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nlp
import json
import time

# Constants
DB_FAISS_PATH = "/home/ubuntu/BITS/SEM_3/ConvAI/llm_and_rag/diff_model/vectorstore/db_faiss"
MODEL_PATH = "/home/ubuntu/BITS/SEM_3/ConvAI/llm_and_rag/diff_model/model/gemma-2-2b-it-Q6_K.gguf"

custom_prompt_template = """
<|context|>
You are an AI assistant that follows instruction extremely well.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

# Function to get doctor availability from SQLite
def get_doctor_availability(doctor_name, date):
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()

    # Get doctor ID based on the name
    cursor.execute('SELECT id FROM doctors WHERE name = ?', (doctor_name,))
    doctor_id = cursor.fetchone()
    
    if doctor_id:
        doctor_id = doctor_id[0]
        # Get available time slots for the doctor on the selected date
        cursor.execute('''
            SELECT time_slot FROM availability
            WHERE doctor_id = ? AND date = ?
        ''', (doctor_id, date))
        slots = cursor.fetchall()
        return [slot[0] for slot in slots]
    
    conn.close()
    return []

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm_model():
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.3,
        max_tokens=2048,
        top_p=1
    )

def create_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    return (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def get_doctors():
    """Retrieve all doctor names from the database."""
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM doctors')
    doctors = cursor.fetchall()
    conn.close()
    return [doctor[0] for doctor in doctors]

def get_available_dates(doctor_name):
    """Retrieve all unique dates for a given doctor from the database."""
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()

    # Get doctor ID based on the name
    cursor.execute('SELECT id FROM doctors WHERE name = ?', (doctor_name,))
    doctor_id = cursor.fetchone()

    if doctor_id:
        doctor_id = doctor_id[0]
        # Get unique dates for the doctor
        cursor.execute('''
            SELECT DISTINCT date FROM availability
            WHERE doctor_id = ?
        ''', (doctor_id,))
        dates = cursor.fetchall()
        conn.close()
        return [date[0] for date in dates]

    conn.close()
    return []

def book_appointment(doctor_name, date, time_slot, patient_name, mobile_number):
    """Book an appointment and mark the slot as booked."""
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()

    # Get doctor ID based on the name
    cursor.execute('SELECT id FROM doctors WHERE name = ?', (doctor_name,))
    doctor_id = cursor.fetchone()

    if doctor_id:
        doctor_id = doctor_id[0]

        # Check if the slot is still available
        cursor.execute('''
            SELECT booked FROM availability
            WHERE doctor_id = ? AND date = ? AND time_slot = ? AND booked = 0
        ''', (doctor_id, date, time_slot))
        available = cursor.fetchone()

        if available:
            # Mark the slot as booked
            cursor.execute('''
                UPDATE availability
                SET booked = 1
                WHERE doctor_id = ? AND date = ? AND time_slot = ?
            ''', (doctor_id, date, time_slot))

            # Insert the appointment record
            cursor.execute('''
                INSERT INTO appointments (doctor_id, date, time_slot, patient_name, mobile_number)
                VALUES (?, ?, ?, ?, ?)
            ''', (doctor_id, date, time_slot, patient_name, mobile_number))

            conn.commit()
            conn.close()
            return True  # Booking successful
        else:
            conn.close()
            return False  # Slot already booked

    conn.close()
    return False  # Doctor not found

# Function to display messages
def display_messages(response):
        st.session_state.messages.append({"role": "assistant", "content": response })
        with st.chat_message("assistant"):
            st.markdown(response)

def collect_patient_details():
    # Ask for patient details
    patient_name = st.text_input("Please enter the patient's name:")
    mobile_number = st.text_input("Please enter the mobile number:")

    # Confirm details button
    confirm_details = st.button("Confirm Details")

    if confirm_details:
        if not patient_name.strip():
            st.error("Patient name cannot be empty.")
        elif len(mobile_number) != 10 or not mobile_number.isdigit():
            st.error("Please enter a valid 10-digit mobile number.")
        else:
            details_collected = True
            st.success("Patient details collected successfully. Proceeding to book the appointment...")

def book_mode():
    st.title("Book a Doctor Appointment")
    st.write("Select a doctor and a time slot from the options below:")

    # Initialize session state variables for patient details
    if 'patient_name' not in st.session_state:
        st.session_state.patient_name = ''
    if 'mobile_number' not in st.session_state:
        st.session_state.mobile_number = ''

    # Input fields for user details
    st.session_state.patient_name = st.text_input("Patient Name", st.session_state.patient_name)
    st.session_state.mobile_number = st.text_input("Mobile Number", st.session_state.mobile_number)

    if len(st.session_state.mobile_number) != 10 or not st.session_state.mobile_number.isdigit():
        st.error("Please enter a valid 10-digit mobile number.")
        return

    # Retrieve doctors dynamically
    doctors = get_doctors()

    # Pre-select doctor if available in session state
    doctor = st.session_state.selected_doctor if st.session_state.selected_doctor in doctors else doctors[0]
    selected_doctor = st.selectbox("Select Doctor:", doctors, index=doctors.index(doctor))

    if selected_doctor:
        # Retrieve available dates dynamically for the selected doctor
        dates = get_available_dates(selected_doctor)

        # Pre-select day if available in session state
        day = st.session_state.selected_day if st.session_state.selected_day in dates else dates[0]
        selected_day = st.selectbox("Select Date:", dates, index=dates.index(day))

        if selected_day:
            # Get available time slots for the selected doctor and date
            slots = get_doctor_availability(selected_doctor, selected_day)

            # Pre-select time slot if available in session state
            time_slot = (
                st.session_state.selected_time_slot
                if st.session_state.selected_time_slot in slots
                else (slots[0] if slots else None)
            )
            if slots:
                selected_time_slot = st.selectbox("Select Time Slot:", slots, index=slots.index(time_slot))

                if st.button("Book Appointment"):
                    success = book_appointment(
                        selected_doctor,
                        selected_day,
                        selected_time_slot,
                        st.session_state.patient_name,
                        st.session_state.mobile_number,
                    )
                    if success:
                        st.success(f"Appointment booked with {selected_doctor} on {selected_day} at {selected_time_slot}.")
                        st.success("Getting you back to chatbot...")
                        
                        # Clear session state for booking inputs
                        st.session_state.selected_doctor = None
                        st.session_state.selected_day = None
                        st.session_state.selected_time_slot = None
                        
                        # Delay before switching to Chat mode
                        time.sleep(2)
                        st.session_state.mode = "Chat"
                        st.rerun()
                    else:
                        st.error("Selected time slot is no longer available. Please choose another.")
            else:
                st.error("No available time slots for the selected doctor on this date.")
    # "Back" button to return to Chat mode
    if st.button("Back to Chat"):
        st.session_state.mode = "Chat"
        st.session_state.selected_doctor = None
        st.session_state.selected_day = None
        st.session_state.selected_time_slot = None
        st.session_state.patient_name = ''
        st.session_state.mobile_number = ''
        st.rerun()


def main():
    st.set_page_config(page_title="Tiny Toes Clinic 👣👶🏻")

    patient_name = ""
    mobile_number = ""
    details_collected = False
    st.session_state.need_to_book = False

    # Initialize session state for mode
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat"  # Default mode is Chat

    # Sidebar
    with st.sidebar:
        
        st.title("Tiny Toes Medical Chatbot 🚀🤖")
        #st.markdown("""## About
        #This chatbot uses an optimized model and a FAISS vector store for medical knowledge.""")
        #st.markdown("### 🔄Bot evolving, stay tuned!")
        #mode = st.radio("Select Mode", ["Chat", "Book Appointment"])
        st.markdown("Welcome to our AI-powered healthcare chatbot!")
        #st.image("path_to_logo.png", width=200)  # Add your logo here
        st.markdown("### Quick Links")
        st.markdown("- [FAQ](#)")
        st.markdown("- [Contact Support](#)")
        
        mode = st.radio(
            "Select Mode",
            ["Chat", "Book Appointment"],
            index=0 if st.session_state.mode == "Chat" else 1,
            key="mode_radio",
        )

    if st.session_state.mode == "Chat":
        st.title("Tiny Toes Clinic 👣👶🏻")
        st.markdown("""<style>
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 450px;
                overflow-y: scroll;
                padding: 15px;
                background-color: #f0f0f5;
                border-radius: 10px;
            }
            .user-bubble {
                background-color: #007bff;
                color: white;
                align-self: flex-end;
                border-radius: 10px;
                padding: 8px;
                margin: 5px;
                max-width: 70%;
                word-wrap: break-word;
            }
            .bot-bubble {
                background-color: #4CAF50;
                color: white;
                align-self: flex-start;
                border-radius: 10px;
                padding: 10px;
                margin: 8px;
                max-width: 70%;
                word-wrap: break-word;
            }
            
        </style>""", unsafe_allow_html=True)

        # Load resources
        st.write("Ask any medical related questions.")
        vectorstore = load_vectorstore()
        llm = load_llm_model()
        rag_chain = create_rag_chain(vectorstore, llm)

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # Create a chat input field
        if prompt := st.chat_input("Ask your question here:"):
            # Store and display the current prompt
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
             # Process the prompt for intent and entity recognition
            processed_data = nlp.classifyText(prompt)
            #st.success(f"Processed Data: {processed_data}")

            # If processed_data is a string, convert it to a dictionary
            if isinstance(processed_data, str):
                try:
                    processed_data = json.loads(processed_data)  # Parse the string into a dictionary
                except json.JSONDecodeError:
                    st.error("Error: Processed data is not in valid JSON format.")
                    return

            # Extract intent and entities from the processed data
            intent = processed_data.get("intent", "")
            #st.success(f"Intent: {intent}")
            #entities = processed_data.get("entities", {})
            #st.success(f"Entity: {entities}")

            # Save intent and entities in session state
            st.session_state.intent = intent
            #st.session_state.entities = entities

            if st.session_state.intent:
                # Handle booking intent and process extracted entities
                if st.session_state.intent == "make appointment":
                    response = f"Intent detected: {intent}."

                    # Extract the entities from session state and process the booking logic
                    #person = entities.get("person", [])
                    # Extract the doctor name correctly
                    #st.session_state.doctor_name_list = processed_data.get("person", [])
                    #st.session_state.doctor_name = doctor_name_list[0].upper() if doctor_name_list else ""
                    st.session_state.selected_doctor = processed_data.get("person", [""])[0].upper() if processed_data.get("person", []) else ""
                    #st.success(doctor_name)
                    #doctor_name = processed_data.get("person", [0])  # Doctor name
                    st.session_state.date = processed_data.get("date0", "")         # Appointment date
                    st.session_state.selected_time_slot = processed_data.get("time0", "")    # Time slot
                    #day = entities.get("day0", "")
                    st.session_state.selected_day = processed_data.get("day0", "")
                    #st.success(f"Day: {day}")

                    # Retrieve doctor names
                    doctors = get_doctors()

                    # Convert the list of doctor names into a readable string
                    doctor_list = ", ".join(doctors)

                    st.session_state.need_to_book = True
                    st.session_state.mode = "Book Appointment"  # Switch mode
                    # Pass detected entities to book_mode
                    book_mode()
                    #book_mode(
                    #    doctor=st.session_state.doctor_name, 
                    #    day=st.session_state.day, 
                    #    time_slot=st.session_state.time_slot
                    #)

                else:
                    st.session_state.messages.append({"role": "assistant", "content": f"Intent detected: {intent}. However, it's not related to booking appointments."})
            else:

                # Generate a response using the RAG chain
                with st.spinner("Processing your question..."):
                    try:
                        response = rag_chain.invoke(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        #if st.session_state.need_to_book:
        #   book_mode()

        # Clear chat button
        if st.session_state.messages:
            if st.button("Clear Chat"):
                st.session_state.messages = []

    elif st.session_state.mode == "Book Appointment":

        if "selected_doctor" not in st.session_state:
            st.session_state.selected_doctor = None
        if "selected_day" not in st.session_state:
            st.session_state.selected_day = None
        if "selected_time_slot" not in st.session_state:
            st.session_state.selected_time_slot = None
        book_mode()
        #book_mode()
                    
if __name__ == "__main__":
    main()
