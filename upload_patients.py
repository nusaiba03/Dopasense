import pandas as pd
import mysql.connector

# Load your CSV
df = pd.read_csv("parkinsons_disease_data.csv")

# Connect to your MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="dopasense_app",
    password="qwertyuiop",
    database="DopaSense"
)

cursor = conn.cursor()

# Optional: Clear the table first (only if you want a fresh start)
cursor.execute("DELETE FROM patients")
conn.commit()

# Insert each row into the database
for index, row in df.iterrows():
    sql = """
    INSERT INTO patients (age, bmi, hypertension, diabetes, moca, functional_assessment)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (
        int(row['Age']),
        float(row['BMI']),
        int(row['Hypertension']),
        int(row['Diabetes']),
        int(row['MoCA']),
        int(row['FunctionalAssessment'])
    )
    cursor.execute(sql, values)

conn.commit()
cursor.close()
conn.close()

print("✅ Upload complete!")
