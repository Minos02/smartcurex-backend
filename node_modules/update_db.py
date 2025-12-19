import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Add missing columns
try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN patient_name TEXT")
    print("✅ Added patient_name column")
except:
    print("⚠️ patient_name already exists")

try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN patient_age TEXT")
    print("✅ Added patient_age column")
except:
    print("⚠️ patient_age already exists")

try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN image_filename TEXT")
    print("✅ Added image_filename column")
except:
    print("⚠️ image_filename already exists")

conn.commit()

# Show final schema
cursor.execute("PRAGMA table_info(predictions)")
print("\n📋 Final predictions table schema:")
for row in cursor.fetchall():
    print(f"   {row[1]:<20} {row[2]:<10}")

conn.close()
print("\n✅ Database schema updated!")
