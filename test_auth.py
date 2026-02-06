"""
Quick test script to verify authentication system setup
"""

print("🔍 Checking authentication system setup...\n")

# Test imports
try:
    from database import init_db, User, OTP
    print("✅ Database module imported successfully")
except Exception as e:
    print(f"❌ Database import failed: {e}")

try:
    from auth import hash_password, verify_password, generate_otp, create_access_token
    print("✅ Auth module imported successfully")
except Exception as e:
    print(f"❌ Auth import failed: {e}")

try:
    from backend import app
    print("✅ Backend module imported successfully")
except Exception as e:
    print(f"❌ Backend import failed: {e}")

# Test database initialization
try:
    init_db()
    print("✅ Database initialized successfully")
    print("   → Database file: users.db")
except Exception as e:
    print(f"❌ Database initialization failed: {e}")

# Test password hashing
try:
    password = "test1234"
    hashed = hash_password(password)
    is_valid = verify_password(password, hashed)
    if is_valid:
        print("✅ Password hashing works correctly")
    else:
        print("❌ Password verification failed")
except Exception as e:
    print(f"❌ Password hashing test failed: {e}")

# Test OTP generation
try:
    otp = generate_otp()
    if len(otp) == 6 and otp.isdigit():
        print(f"✅ OTP generation works (sample: {otp})")
    else:
        print("❌ OTP format incorrect")
except Exception as e:
    print(f"❌ OTP generation failed: {e}")

# Test JWT token
try:
    token = create_access_token({"sub": "test@example.com"})
    if token:
        print("✅ JWT token creation works")
    else:
        print("❌ JWT token creation failed")
except Exception as e:
    print(f"❌ JWT token test failed: {e}")

# Check static files
import os
files_to_check = [
    "static/login.html",
    "static/signup.html",
    "static/index.html"
]

print("\n📄 Checking static files:")
for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} (missing)")

print("\n" + "="*60)
print("🎉 Authentication system check complete!")
print("="*60)
print("\n📚 Next steps:")
print("   1. Run: python backend.py")
print("   2. Visit: http://localhost:8000")
print("   3. Create an account and test the system")
print("\n📖 See AUTH_SETUP.md for detailed documentation")
