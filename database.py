"""
Database models and configuration for authentication
"""
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# SQLite database
DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OTP(Base):
    """OTP model for verification"""
    __tablename__ = "otps"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    phone = Column(String, index=True, nullable=False)
    otp_code = Column(String, nullable=False)
    is_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


class ActivityLog(Base):
    """Activity log model for tracking user actions"""
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)  # Nullable for failed login attempts
    email = Column(String, index=True, nullable=True)
    activity_type = Column(String, index=True, nullable=False)  # LOGIN, LOGOUT, SIGNUP, OTP_VERIFY, etc.
    description = Column(String, nullable=False)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    status = Column(String, nullable=False)  # SUCCESS, FAILED, ERROR
    extra_data = Column(String, nullable=True)  # JSON string for additional data
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_activity(
    db,
    activity_type: str,
    description: str,
    status: str = "SUCCESS",
    user_id: int = None,
    email: str = None,
    ip_address: str = None,
    user_agent: str = None,
    extra_data: str = None
):
    """
    Log user activity to database
    
    Args:
        db: Database session
        activity_type: Type of activity (LOGIN, LOGOUT, SIGNUP, etc.)
        description: Description of the activity
        status: Status of activity (SUCCESS, FAILED, ERROR)
        user_id: User ID if available
        email: User email
        ip_address: User's IP address
        user_agent: Browser/client user agent
        extra_data: Additional data as JSON string
    """
    try:
        log_entry = ActivityLog(
            user_id=user_id,
            email=email,
            activity_type=activity_type,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            extra_data=extra_data
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"Failed to log activity: {str(e)}")
        db.rollback()
