DopaSense: Parkinson’s Readmission Prediction App

This is a Streamlit-based web app for clinicians to predict hospital readmission risk for Parkinson’s patients, view patient data, and manage user accounts.

Getting Started:

Install Python 3.8+ if not already installed.
Install required Python packages by running:
pip install streamlit pandas numpy scikit-learn imbalanced-learn mysql-connector-python bcrypt

Set up MySQL:
Create the database and tables using the SQL below (you can run this in MySQL Workbench or via CLI):

CREATE DATABASE IF NOT EXISTS DopaSense;
USE DopaSense;

CREATE USER IF NOT EXISTS 'dopasense_app'@'localhost' IDENTIFIED BY 'qwertyuiop'; 
GRANT ALL PRIVILEGES ON DopaSense.* TO 'dopasense_app'@'localhost';
FLUSH PRIVILEGES;

CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    specialization VARCHAR(100),
    email VARCHAR(100)
);

INSERT INTO users (username, password_hash, name, specialization, email)
VALUES ('admin', '$2b$12$duMekjBc1eHcJkjKq2GKsOsAg8RzdDKQh2gcFAXDgwvXjEeb717rO', 'John', 'Neurology', 'admin@example.com');

CREATE TABLE IF NOT EXISTS patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL,
    bmi FLOAT,
    hypertension TINYINT(1),
    diabetes TINYINT(1),
    moca INT,
    functional_assessment INT
);

CREATE TABLE IF NOT EXISTS readmission_predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    predicted_label TINYINT(1),
    prediction_probability FLOAT,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

Clone the repository or copy the project files to your local machine.

Prepare your dataset:
Place a parkinsons_disease_data.csv file with relevant columns in the same directory.

Run the app:
streamlit run dopasense.py

Default Login:
Username: admin
Password: adminpass
