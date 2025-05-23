import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import mysql.connector
import bcrypt

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# MySQL connection function
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="dopasense_app",
        password="qwertyuiop",
        database="DopaSense"
    )

# Prepare data function
@st.cache_data
def prepare_data():
    df = pd.read_csv("parkinsons_disease_data.csv")

    # Ensure the target column exists
    if "Diagnosis" not in df.columns:
        st.error("Error: The 'Diagnosis' column is missing from the dataset!")
        return None, None, None, None

    # Handle missing values
    df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)

    # Add synthetic symptom features
    np.random.seed(42)
    if 'Tremors' not in df.columns:
        df['Tremors'] = np.random.randint(0, 11, size=len(df))
    if 'Sleep' not in df.columns:
        df['Sleep'] = np.random.randint(0, 11, size=len(df))
    if 'Mood' not in df.columns:
        df['Mood'] = np.random.randint(0, 11, size=len(df))

    # Define features and target
    relevant_features = ["Age", "BMI", "Hypertension", "Diabetes", "MoCA", "FunctionalAssessment", "Tremors", "Sleep", "Mood"]
    if not all(feature in df.columns for feature in relevant_features):
        st.error("Error: Some required features are missing from the dataset!")
        return None, None, None, None

    X = df[relevant_features]
    y = df["Diagnosis"]

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Model function
@st.cache_resource
def train_model(X_train, y_train):
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Evaluate Model function
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions), precision_score(y_test, predictions, zero_division=0), recall_score(y_test, predictions, zero_division=0), f1_score(y_test, predictions, zero_division=0), classification_report(y_test, predictions, zero_division=0)

# Register page
def register_page():
    st.title("Register")
    st.markdown("Create a new account for DopaSense platform.")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    specialization = st.text_input("Specialization")
    
    if st.button("Register"):
        if username and password and name and email and specialization:
            # Hash the password using bcrypt
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Connect to the database
            conn = get_connection()
            cursor = conn.cursor()

            # Check if username already exists
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                st.error("Username already exists. Please choose a different one.")
            else:
                # Insert new user into the database
                insert_query = """
                    INSERT INTO users (username, password_hash, name, specialization, email)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (username, password_hash, name, specialization, email))
                conn.commit()
                cursor.close()
                conn.close()

                st.success("Account created successfully! Please log in.")
                if  st.button("Go to Login Page"):
                    st.session_state.page = "Login"
                    st.experimental_rerun()

        else:
            st.warning("Please fill in all the fields.")

# Login page
def login_page():
    st.title("Login")
    st.markdown("Welcome back! Please log in to access the DopaSense platform.")
    
    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username and password:
            # Connect to the database
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)

            # Check if the user exists
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()

            cursor.close()
            conn.close()

            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                # Store session info
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(f"Welcome, {user['name']}!")
                st.rerun()  # Re-run to refresh and show homepage
            else:
                st.error("Invalid username or password.")
        else:
            st.warning("Please enter both username and password.")


# Homepage
def homepage():
    st.title("👋 Welcome to DopaSense")

    # Welcome callout
    st.info("""
    👩‍⚕️ **DopaSense** empowers clinicians with predictive insights for managing Parkinson’s disease patients.  
    Use this platform to make smarter, data-driven decisions that enhance patient care.
    """)

    st.markdown("### 🔍 What You Can Do")

    # Feature highlights
    st.markdown("""
    - 📈 **Predict Readmission Risk**  
      Use patient data to estimate the probability of hospital readmission and adjust care plans accordingly.
      
    - 📊 **Explore Patient Trends**  
      View patient demographics and progression patterns with visual analytics.

    - 🧠 **Learn About Parkinson’s**  
      Access quick-reference insights and support materials for understanding PD.

    - 🗂 **Manage Your Profile**  
      Update your contact info and track usage right from the platform.

    """)

    st.markdown("---")

# Parkinson’s Info Page
def parkinsons_info_page():
    st.title("Parkinson’s Disease Information and Analytics")
    st.markdown("""
    Parkinson’s disease (PD) is a progressive neurological disorder that affects movement and coordination. Understanding the key metrics of Parkinson’s can help healthcare providers deliver better care.
    """)

    st.subheader("Key Insights")
    st.write("""
    - **Ages Affected**: Primarily 50+ but early-onset cases exist.
    - **Common Symptoms**: Tremors, rigidity, bradykinesia, postural instability.
    - **Treatment Approaches**: Medications (e.g., Levodopa), physical therapy, and lifestyle adjustments.
    """)

    st.subheader("Patient Trends")
    df = pd.read_csv("parkinsons_disease_data.csv")

    st.write("### Data Overview")
    st.dataframe(df.head(10))

    st.write("### Age Distribution of Patients")
    age_counts = df['Age'].value_counts().sort_index()
    st.bar_chart(age_counts, use_container_width=True)
    st.caption("**X-axis:** Age (years) | **Y-axis:** Number of Patients")

    st.write("### Disease Progression Over Time")
    if 'Age' in df.columns and 'UPDRS' in df.columns:
        st.line_chart(df.set_index('Age')[['UPDRS']])
        st.caption("**X-axis:** Age (years) | **Y-axis:** Unified Parkinson’s Disease Rating Scale (UPDRS) Score")
    else:
        st.warning("Required columns for disease progression visualization ('Age', 'UPDRS') are missing from the dataset.")
  
# User Profile Page
def user_profile():
    st.title("User Profile")

    if 'user' not in st.session_state:
        st.error("You must be logged in to view this page.")
        return

    user = st.session_state.user

    st.markdown(f"""
    **Name**: {user['name']}  
    **Specialization**: {user['specialization']}  
    """)

    # Editable email field
    updated_email = st.text_input("Email", user['email'])

    # Notes field (optional)
    notes = st.text_area("Add Notes", placeholder="Add any notes or preferences here.")

    if st.button("Save Changes"):
        conn = get_connection()
        cursor = conn.cursor()
        update_query = "UPDATE users SET email = %s WHERE user_id = %s"
        cursor.execute(update_query, (updated_email, user['user_id']))
        conn.commit()
        cursor.close()
        conn.close()

        # Update session state
        st.session_state.user['email'] = updated_email
        st.success("Profile updated successfully!")

        
def patient_medical_history():
    st.title("Patient Medical History")
    st.markdown("Search for a specific patient by their ID, or explore filtered data below.")

    df = pd.read_csv("parkinsons_disease_data.csv")

    # Search by Patient ID
    search_id = st.number_input("Enter Patient ID to search", min_value=3058, max_value=5162, step=1)
    if search_id:
        result = df[df["PatientID"] == search_id]
        if not result.empty:
            st.write("### Patient Data")
            st.dataframe(result)
        else:
            st.warning("No patient found with that ID.")

    # Add filter section below search
    st.subheader("Filter Patient Data")
    min_age = st.slider("Minimum Age", 40, 85, 50)
    max_age = st.slider("Maximum Age", 40, 85, 70)
    filtered_df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]
    st.write(f"Showing patients aged between {min_age} and {max_age}")
    st.dataframe(filtered_df)



def predict_readmission():
    st.title("Predict Readmission Risk")

    df = pd.read_csv("parkinsons_disease_data.csv")

    # Search for Patient
    patient_id = st.selectbox("Select Patient ID", df["PatientID"].unique())
    patient_row = df[df["PatientID"] == patient_id].squeeze()

    if patient_row is None or patient_row.empty:
        st.warning("No data found for this patient.")
        return

    st.subheader("Adjust Patient Parameters")

    # Auto-fill sliders with patient data
    age = st.slider("Age", 40, 90, int(patient_row["Age"]))
    bmi = st.slider("BMI", 15.0, 40.0, float(patient_row["BMI"]), step=0.1)
    hypertension = st.radio("Hypertension", [0, 1], index=int(patient_row["Hypertension"]))
    diabetes = st.radio("Diabetes", [0, 1], index=int(patient_row["Diabetes"]))
    moca = st.slider("MoCA Score", 0, 30, int(patient_row["MoCA"]))
    functional_assessment = st.slider("Functional Assessment", 0, 10, int(patient_row["FunctionalAssessment"]))

    # Add clinician input for current symptoms
    tremors = st.slider("Tremors Severity (0 = None, 10 = Severe)", 0, 10, 5)
    sleep = st.slider("Sleep Disturbance (0 = None, 10 = Severe)", 0, 10, 5)
    mood = st.slider("Mood Changes (0 = Stable, 10 = Unstable)", 0, 10, 5)

    # Prepare model input
    patient_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Hypertension": [hypertension],
        "Diabetes": [diabetes],
        "MoCA": [moca],
        "FunctionalAssessment": [functional_assessment],
        "Tremors": [tremors],
        "Sleep": [sleep],
        "Mood": [mood]
    })

    # Prepare and train model
    X_train, X_test, y_train, y_test = prepare_data()
    if X_train is None:
        return

    model = train_model(X_train, y_train)

    # Make prediction
    prediction = model.predict(patient_data)
    prediction_prob = model.predict_proba(patient_data)[0][1]

    # Show result
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"Prediction: **Readmitted** (Risk: {prediction_prob:.2%})")
    else:
        st.success(f"Prediction: **Not Readmitted** (Risk: {prediction_prob:.2%})")

    # Print evaluation metrics to terminal
    accuracy, precision, recall, f1, report = evaluate_model(model, X_test, y_test)

    print("### Model Evaluation on Test Data ###")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(report)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main function
def main():
    load_css("styles/hospital.css")
    # Check if the user is logged in
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        # If not logged in, show Login or Register options
        page = st.sidebar.selectbox("Choose an action", ["Login", "Register"])
        
        if page == "Login":
            login_page()
        elif page == "Register":
            register_page()
    else:
        # logo
        st.sidebar.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
        st.sidebar.image("assets/logo.png", width=180)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        # If logged in, show the homepage and other pages
        st.sidebar.title("Navigation")
        st.sidebar.markdown(f"Logged in as: **{st.session_state.user['name']}**")
        pages = {
            "Homepage": homepage,
            "Parkinson’s Info & Analytics": parkinsons_info_page,
            "User Profile": user_profile,
            "Patient Medical History": patient_medical_history,
            "Predict Readmission": predict_readmission
        }
        choice = st.sidebar.radio("Go to", list(pages.keys()))
        pages[choice]()

if __name__ == "__main__":
    main()
