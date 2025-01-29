import streamlit as st
import sqlite3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from operator import itemgetter
import nlp
import json
import time
import torch

# Constants
DB_FAISS_PATH = "/home/ubuntu/BITS/SEM_3/ConvAI/llm_and_rag/diff_model/vectorstore/db_faiss"
MODEL_PATH = "/home/ubuntu/BITS/SEM_3/ConvAI/llm_and_rag/diff_model/model/gemma-2-2b-it-Q6_K.gguf"

def format_history(messages):
    """Formats chat history into a string for the prompt"""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            history.append(f"Assistant: {msg['content']}")
    return "\n".join(history)

custom_prompt_template = """
<|context|>
You are a medical AI assistant for Tiny Toes Clinic. Your role is to answer healthcare questions, provide child care advice, and assist with appointments. 
**Important Rules:**
1. Only respond to medical or appointment-related queries.
2. If asked about non-medical topics (e.g., weather, general knowledge), politely decline.
3. Be truthful and use the context below when relevant.
Previous conversation:
{history}

Relevant Context:
{context}
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
        n_ctx=2048,
        top_p=1
    )

def create_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    return (
        RunnableParallel({
            "context": itemgetter("query") | retriever,
            "query": itemgetter("query"),
            "history": itemgetter("history")
        })
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
    st.session_state.need_to_book = False

def get_available_doctors(selected_day):
    """Retrieve doctors available on a specific day."""
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()

    # Get all doctors and their available time slots for the selected day
    cursor.execute('''
        SELECT d.name, a.time_slot
        FROM doctors d
        JOIN availability a ON d.id = a.doctor_id
        WHERE a.date = ? AND a.booked = 0
    ''', (selected_day,))
    available_doctors = cursor.fetchall()
    conn.close()

    # Organize the data into a dictionary: {doctor: [time_slots]}
    doctor_schedule = {}
    for doctor, time_slot in available_doctors:
        if doctor not in doctor_schedule:
            doctor_schedule[doctor] = []
        doctor_schedule[doctor].append(time_slot)
    
    return doctor_schedule

# Modify the sidebar to display a dropdown for selecting the day
def display_day_selector():
    st.sidebar.title("Check Doctor Availability")
    
    # Get unique available dates from the database
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT date FROM availability WHERE booked = 0')
    available_dates = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Dropdown to select a day
    selected_day = st.sidebar.selectbox("Select Day", available_dates)

    # Display available doctors and their time slots for the selected day
    if selected_day:
        st.sidebar.markdown(f"**Available Doctors on {selected_day}:**")
        doctor_schedule = get_available_doctors(selected_day)
        if doctor_schedule:
            for doctor, time_slots in doctor_schedule.items():
                st.sidebar.markdown(f"**Doctor:** {doctor}")
                st.sidebar.markdown(f"**Time Slots:** {', '.join(time_slots)}")
                st.sidebar.markdown("---")
        else:
            st.sidebar.markdown("No doctors available on this day.")

# Add a function to get doctor schedules
def get_doctor_schedule(doctor_name):
    """Retrieve the schedule of a specific doctor."""
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()

    # Get doctor ID based on the name
    cursor.execute('SELECT id FROM doctors WHERE name = ?', (doctor_name,))
    doctor_id = cursor.fetchone()

    if doctor_id:
        doctor_id = doctor_id[0]
        # Get all available time slots for the doctor
        cursor.execute('''
            SELECT date, time_slot FROM availability
            WHERE doctor_id = ? AND booked = 0
        ''', (doctor_id,))
        schedule = cursor.fetchall()
        conn.close()
        return schedule

    conn.close()
    return []

# Modify the sidebar to display doctor schedules
def display_doctor_schedules():
    st.sidebar.title("Doctor Schedules")
    doctors = get_doctors()
    for doctor in doctors:
        st.sidebar.markdown(f"**Doctor:** {doctor}")
        schedule = get_doctor_schedule(doctor)
        if schedule:
            for date, time_slot in schedule:
                st.sidebar.markdown(f"- **Date:** {date}, **Time:** {time_slot}")
        else:
            st.sidebar.markdown("No available slots.")
        st.sidebar.markdown("---")


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
        st.markdown("Welcome to our AI-powered healthcare chatbot!")
        st.markdown("### Quick Links")
        st.markdown("- [FAQ](#)")
        st.markdown("- [Contact Support](#)")

        st.session_state.mode = st.radio(
            "Select Mode",
            ["Chat", "Book Appointment"],
            index=0 if st.session_state.mode == "Chat" else 1,
            key="mode_radio",
        )
        
        # Display day selector and doctor availability
        display_day_selector()

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

                #if not st.session_state.get('need_to_book', False):
                if st.session_state.need_to_book == False:

                    history_messages = st.session_state.messages[:-1]
                    history_str = format_history(history_messages)
                    
                    # Prepare input with history
                    input_dict = {
                        "query": prompt,
                        "history": history_str
                    }

                    # Generate response
                    with st.spinner("Processing your question..."):
                        try:
                            response = rag_chain.invoke(input_dict)
                            display_messages(response)
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
