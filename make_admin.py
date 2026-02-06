"""
Script to make a user admin
"""
import sqlite3
import sys

def make_admin(email):
    """Make a user admin by email"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id, email, is_admin FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            print(f"❌ User with email '{email}' not found!")
            conn.close()
            return False
        
        user_id, user_email, is_admin = user
        
        if is_admin:
            print(f"✓ User '{user_email}' is already an admin!")
            conn.close()
            return True
        
        # Make user admin
        cursor.execute("UPDATE users SET is_admin = 1 WHERE email = ?", (email,))
        conn.commit()
        
        print(f"✅ Successfully made '{user_email}' an admin!")
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_admin.py <email>")
        print("Example: python make_admin.py admin@example.com")
        sys.exit(1)
    
    email = sys.argv[1]
    make_admin(email)
