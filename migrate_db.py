"""
Database migration script to add is_admin column and create admin user
"""
import sqlite3
from auth import hash_password

def migrate_database():
    """Add is_admin column to existing database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # Check if is_admin column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_admin' not in columns:
            print("Adding is_admin column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
            conn.commit()
            print("✅ is_admin column added successfully!")
        else:
            print("✓ is_admin column already exists")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {str(e)}")
        conn.close()
        return False

def create_admin_user():
    """Create admin user with email admin@gmail.com and password 12345678"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # Check if admin user exists
        cursor.execute("SELECT id, email, is_admin FROM users WHERE email = ?", ('admin@gmail.com',))
        user = cursor.fetchone()
        
        if user:
            user_id, email, is_admin = user
            if not is_admin:
                # Make existing user admin
                cursor.execute("UPDATE users SET is_admin = 1 WHERE email = ?", ('admin@gmail.com',))
                conn.commit()
                print(f"✅ Updated existing user '{email}' to admin!")
            else:
                print(f"✓ User '{email}' is already an admin!")
        else:
            # Create new admin user
            hashed_pw = hash_password('12345678')
            cursor.execute("""
                INSERT INTO users (email, phone, hashed_password, is_verified, is_admin, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, ('admin@gmail.com', '+1234567890', hashed_pw, 1, 1))
            conn.commit()
            print("✅ Created new admin user: admin@gmail.com / 12345678")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin: {str(e)}")
        conn.close()
        return False

if __name__ == "__main__":
    print("🔄 Running database migration...")
    print()
    
    # Step 1: Add is_admin column
    if migrate_database():
        print()
        # Step 2: Create admin user
        create_admin_user()
        print()
        print("=" * 50)
        print("✅ Migration completed successfully!")
        print()
        print("Admin Login Credentials:")
        print("  Email: admin@gmail.com")
        print("  Password: 12345678")
        print()
        print("Access admin panel at: http://localhost:8000/admin")
        print("=" * 50)
    else:
        print("❌ Migration failed!")
