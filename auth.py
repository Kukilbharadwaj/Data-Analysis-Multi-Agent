"""
Authentication utilities for password hashing, JWT tokens, and OTP generation
"""
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
import random
import string
from typing import Optional
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Security configuration
SECRET_KEY = "your-secret-key-change-this-in-production-use-env-variable"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Email configuration (Gmail SMTP - Free)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")  # Your Gmail address
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # Your Gmail app password


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    # Convert password to bytes (bcrypt works with bytes)
    # Bcrypt has a maximum password length of 72 bytes
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string for storage
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    try:
        # Convert password to bytes
        password_bytes = plain_password.encode('utf-8')
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        
        # Convert hash to bytes if it's a string
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')
        
        # Verify
        return bcrypt.checkpw(password_bytes, hashed_password)
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_otp(length: int = 6) -> str:
    """Generate random OTP code"""
    return ''.join(random.choices(string.digits, k=length))


def is_otp_expired(created_at: datetime, expires_at: datetime) -> bool:
    """Check if OTP is expired"""
    return datetime.utcnow() > expires_at


def send_otp_email(email: str, otp: str) -> bool:
    """
    Send OTP via email using Gmail SMTP (Free)
    """
    # If SMTP not configured, fall back to console printing
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print(f"\n{'='*60}")
        print(f"📧 [DEV MODE - Email Not Configured]")
        print(f"{'='*60}")
        print(f"OTP for {email}: {otp}")
        print(f"\nℹ️  To send real emails:")
        print(f"   1. Open .env file")
        print(f"   2. Set SMTP_EMAIL to your Gmail address")
        print(f"   3. Set SMTP_PASSWORD to your Gmail App Password")
        print(f"   4. See GMAIL_SETUP_GUIDE.md for detailed instructions")
        print(f"{'='*60}\n")
        return True
    
    try:
        print(f"📧 Attempting to send OTP email to {email}...")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Your OTP Code - {otp}'
        msg['From'] = SMTP_EMAIL
        msg['To'] = email
        
        # HTML email body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .otp-code {{ background: #667eea; color: white; font-size: 32px; 
                            font-weight: bold; padding: 20px; text-align: center; 
                            border-radius: 8px; letter-spacing: 8px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 Multi-Agent AI Data Analyst</h1>
                    <p>Email Verification</p>
                </div>
                <div class="content">
                    <h2>Your Verification Code</h2>
                    <p>Thank you for signing up! Please use the following OTP code to verify your email address:</p>
                    <div class="otp-code">{otp}</div>
                    <p><strong>This code will expire in 10 minutes.</strong></p>
                    <p>If you didn't request this code, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>This is an automated message, please do not reply.</p>
                    <p>&copy; 2026 Multi-Agent AI Data Analyst Platform</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text alternative
        text_body = f"""
        Multi-Agent AI Data Analyst - Email Verification
        
        Your OTP Code: {otp}
        
        This code will expire in 10 minutes.
        
        If you didn't request this code, please ignore this email.
        """
        
        # Attach both versions
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        print(f"   Connecting to {SMTP_SERVER}:{SMTP_PORT}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure connection
            print(f"   Logging in as {SMTP_EMAIL}...")
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            print(f"   Sending email...")
            server.send_message(msg)
        
        print(f"✅ OTP email sent successfully to {email}")
        print(f"   Check your inbox (and spam folder if not found)")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"\n❌ SMTP Authentication Failed!")
        print(f"   Error: {str(e)}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Make sure you're using Gmail App Password (NOT regular password)")
        print(f"   2. Enable 2-Step Verification: https://myaccount.google.com/security")
        print(f"   3. Generate App Password: https://myaccount.google.com/apppasswords")
        print(f"   4. Check SMTP_EMAIL and SMTP_PASSWORD in .env file")
        print(f"\n📧 [FALLBACK] OTP for {email}: {otp}")
        return False
        
    except smtplib.SMTPException as e:
        print(f"\n❌ SMTP Error occurred!")
        print(f"   Error: {str(e)}")
        print(f"   Check your internet connection and Gmail settings")
        print(f"\n📧 [FALLBACK] OTP for {email}: {otp}")
        return False
        
    except Exception as e:
        print(f"\n❌ Failed to send email to {email}")
        print(f"   Error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        print(f"\n📧 [FALLBACK] OTP for {email}: {otp}")
        return False


def send_otp_sms(phone: str, otp: str) -> bool:
    """
    Send OTP via SMS (Console print for now - SMS requires paid service)
    For free SMS in production, you can use Twilio trial credits
    """
    print(f"📱 [DEV MODE] SMS OTP for {phone}: {otp}")
    print(f"ℹ️  SMS sending disabled. OTP shown in console for development.")
    # TODO: Implement Twilio trial or other free SMS service if needed
    return True
