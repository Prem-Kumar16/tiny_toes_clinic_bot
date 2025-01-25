import sqlite3

def create_database():
    conn = sqlite3.connect("doctor_appointments.db")
    cursor = conn.cursor()

    # Create tables

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS availability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            date TEXT NOT NULL,
            time_slot TEXT NOT NULL,
            booked INTEGER DEFAULT 0,
            FOREIGN KEY(doctor_id) REFERENCES doctors(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            date TEXT NOT NULL,
            time_slot TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            mobile_number TEXT NOT NULL,
            FOREIGN KEY(doctor_id) REFERENCES doctors(id)
        )
    ''')

    # Insert dummy data for doctors and availability
    doctors = [
        ("PREM",),
        ("PAVI",),
        ("NANDY",),
        ("KRISH",)
    ]

    cursor.executemany('INSERT INTO doctors (name) VALUES (?)', doctors)

    availability_data = [
        (1, "Monday", "10:00"),
        (1, "Monday", "11:00"),
        (2, "Monday", "12:00"),
        (2, "Monday", "13:00"),
        (3, "Monday", "16:00"),
        (3, "Monday", "17:00"),
        (4, "Monday", "18:00"),
        (4, "Monday", "19:00"),
        (1, "Tuesday", "10:00"),
        (1, "Tuesday", "11:00"),
        (2, "Tuesday", "12:00"),
        (2, "Tuesday", "13:00"),
        (3, "Tuesday", "16:00"),
        (3, "Tuesday", "17:00"),
        (4, "Tuesday", "18:00"),
        (4, "Tuesday", "19:00"),
        (1, "Wednesday", "10:00"),
        (1, "Wednesday", "11:00"),
        (2, "Wednesday", "12:00"),
        (2, "Wednesday", "13:00"),
        (3, "Wednesday", "16:00"),
        (3, "Wednesday", "17:00"),
        (4, "Wednesday", "18:00"),
        (4, "Wednesday", "19:00"),
        (1, "Thursday", "10:00"),
        (1, "Thursday", "11:00"),
        (2, "Thursday", "12:00"),
        (2, "Thursday", "13:00"),
        (3, "Thursday", "16:00"),
        (3, "Thursday", "17:00"),
        (4, "Thursday", "18:00"),
        (4, "Thursday", "19:00"),
        (1, "Friday", "10:00"),
        (1, "Friday", "11:00"),
        (2, "Friday", "12:00"),
        (2, "Friday", "13:00"),
        (3, "Friday", "16:00"),
        (3, "Friday", "17:00"),
        (4, "Friday", "18:00"),
        (4, "Friday", "19:00"),
        (1, "Saturday", "10:00"),
        (1, "Saturday", "11:00"),
        (2, "Saturday", "12:00"),
        (2, "Saturday", "13:00"),
        (3, "Sunday", "10:00"),
        (3, "Sunday", "11:00"),
        (4, "Sunday", "12:00"),
        (4, "Sunday", "13:00"),
    ]

    cursor.executemany('INSERT INTO availability (doctor_id, date, time_slot) VALUES (?, ?, ?)', availability_data)

    conn.commit()
    conn.close()

# Uncomment to run this once and create the database
create_database()
