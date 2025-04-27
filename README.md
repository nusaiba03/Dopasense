# Dopasense

Try using google chrome if it doesnt work on safari.
useername: admin
password: adminpass

Database:
CREATE DATABASE IF NOT EXISTS DopaSense;
USE DopaSense;

# Create the app user
CREATE USER IF NOT EXISTS 'dopasense_app'@'localhost' IDENTIFIED BY 'qwertyuiop'; 
GRANT ALL PRIVILEGES ON DopaSense.* TO 'dopasense_app'@'localhost';

#Apply the privilege changes
FLUSH PRIVILEGES;

CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    specialization VARCHAR(100),
    email VARCHAR(100)
);
#username:admin password:adminpass
INSERT INTO users (username, password_hash, name, specialization, email)
VALUES ('admin', '$2b$12$duMekjBc1eHcJkjKq2GKsOsAg8RzdDKQh2gcFAXDgwvXjEeb717rO', 'John', 'Neurology', 'admin@example.com');


CREATE TABLE patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL,
    bmi FLOAT,
    hypertension TINYINT(1),
    diabetes TINYINT(1),
    moca INT,
    functional_assessment INT
);
