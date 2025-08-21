import bcrypt

# Change the password here
password = "adminpass"

# Generate the bcrypt hash
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# Print the hash
print("Hashed password:\n", hashed.decode())
