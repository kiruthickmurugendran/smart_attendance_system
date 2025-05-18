i have usen postgres db for this project

i have created two tables (students , attendance):

query for student table 
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    roll_number TEXT NOT NULL,
    department TEXT NOT NULL,
    embedding FLOAT8[] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


query for attendance table

CREATE TABLE attendance (
    attendance_id SERIAL PRIMARY KEY,
    roll_number TEXT NOT NULL,
    present BOOLEAN NOT NULL DEFAULT TRUE,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    date DATE DEFAULT CURRENT_DATE
);
