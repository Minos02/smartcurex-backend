# Create check_db.py
import sqlite3

conn = sqlite3.connect('smartcurex.db')
cursor = conn.cursor()

# Check if predictions table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
if cursor.fetchone():
    print("✅ predictions table EXISTS")
    
    # Show schema
    cursor.execute("PRAGMA table_info(predictions)")
    print("\n📋 Table Schema:")
    for row in cursor.fetchall():
        print(f"   {row[1]} ({row[2]})")
    
    # Count rows
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    print(f"\n📊 Total predictions: {count}")
    
    # Show last 5 rows
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
    print("\n📝 Last 5 predictions:")
    for row in cursor.fetchall():
        print(f"   {row}")
else:
    print("❌ predictions table NOT FOUND")

conn.close()
