"""
FastAPI Backend for Multi-Agent AI Data Analyst Platform with Authentication
Enhanced with advanced data loading for complex and unstructured data
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Cookie, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import os
import pandas as pd
import asyncio
from typing import AsyncGenerator, Optional
import json
from datetime import datetime, timedelta
from multi_agent_analyst import analyze_dataset_streaming
from database import init_db, get_db, User, OTP, ActivityLog, log_activity
from auth import (
    hash_password, 
    verify_password, 
    create_access_token, 
    verify_token,
    generate_otp,
    is_otp_expired,
    send_otp_email,
    send_otp_sms
)
from data_loader import load_and_process_data

# Initialize database
init_db()

app = FastAPI(title="Multi-Agent AI Data Analyst")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models for request/response
class SignupRequest(BaseModel):
    email: EmailStr
    phone: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str


class ResendOTPRequest(BaseModel):
    email: EmailStr


# Helper function to get current user from token
def get_current_user(token: Optional[str] = Cookie(None, alias="access_token"), db: Session = Depends(get_db)):
    """Get current authenticated user from cookie token"""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user = db.query(User).filter(User.email == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Email not verified")
    
    return user


# Helper function to check if user is admin
def get_current_admin(current_user: User = Depends(get_current_user)):
    """Check if current user is admin"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the login page"""
    html_path = "static/login.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <body>
            <h1>Multi-Agent AI Data Analyst</h1>
            <p>Login page not found. Please create static/login.html</p>
        </body>
    </html>
    """


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(current_user: User = Depends(get_current_user)):
    """Serve the main dashboard (protected)"""
    html_path = "static/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <body>
            <h1>Dashboard</h1>
            <p>Dashboard not found.</p>
        </body>
    </html>
    """


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(current_admin: User = Depends(get_current_admin)):
    """Serve the admin dashboard (protected - admin only)"""
    html_path = "static/admin.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <body>
            <h1>Admin Dashboard</h1>
            <p>Admin dashboard not found.</p>
        </body>
    </html>
    """


@app.get("/signup", response_class=HTMLResponse)
async def signup_page():
    """Serve the signup page"""
    html_path = "static/signup.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <body>
            <h1>Sign Up</h1>
            <p>Signup page not found.</p>
        </body>
    </html>
    """


# Authentication endpoints
@app.post("/api/auth/signup")
async def signup(request: SignupRequest, req: Request, db: Session = Depends(get_db)):
    """Register a new user and send OTP"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == request.email) | (User.phone == request.phone)
        ).first()
        
        if existing_user:
            log_activity(
                db=db,
                activity_type="SIGNUP_FAILED",
                description=f"Signup attempt with existing email/phone: {request.email}",
                status="FAILED",
                email=request.email,
                ip_address=client_ip,
                user_agent=user_agent,
                extra_data=json.dumps({"reason": "email_or_phone_exists"})
            )
            raise HTTPException(status_code=400, detail="Email or phone already registered")
        
        # Validate password strength
        if len(request.password) < 8:
            log_activity(
                db=db,
                activity_type="SIGNUP_FAILED",
                description=f"Signup attempt with weak password: {request.email}",
                status="FAILED",
                email=request.email,
                ip_address=client_ip,
                user_agent=user_agent,
                extra_data=json.dumps({"reason": "weak_password"})
            )
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
        # Create new user
        hashed_pwd = hash_password(request.password)
        new_user = User(
            email=request.email,
            phone=request.phone,
            hashed_password=hashed_pwd,
            is_verified=False
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Generate and save OTP
        otp_code = generate_otp()
        otp_expires = datetime.utcnow() + timedelta(minutes=10)
        
        new_otp = OTP(
            email=request.email,
            phone=request.phone,
            otp_code=otp_code,
            expires_at=otp_expires
        )
        
        db.add(new_otp)
        db.commit()
        
        # Send OTP
        send_otp_email(request.email, otp_code)
        send_otp_sms(request.phone, otp_code)
        
        # Log successful signup
        log_activity(
            db=db,
            activity_type="SIGNUP",
            description=f"New user registered: {request.email}",
            status="SUCCESS",
            user_id=new_user.id,
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"phone": request.phone, "otp_sent": True})
        )
        
        return {
            "success": True,
            "message": "User registered successfully. Please verify your email with the OTP sent.",
            "email": request.email
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log_activity(
            db=db,
            activity_type="SIGNUP_ERROR",
            description=f"Signup error for {request.email}: {str(e)}",
            status="ERROR",
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"error": str(e)})
        )
        raise HTTPException(status_code=500, detail="Internal server error during signup")


@app.post("/api/auth/verify-otp")
async def verify_otp(request: VerifyOTPRequest, req: Request, db: Session = Depends(get_db)):
    """Verify OTP and activate user account"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    # Find the latest unused OTP for this email
    otp_record = db.query(OTP).filter(
        OTP.email == request.email,
        OTP.is_used == False
    ).order_by(OTP.created_at.desc()).first()
    
    if not otp_record:
        log_activity(
            db=db,
            activity_type="OTP_VERIFY_FAILED",
            description=f"OTP verification failed - No OTP found: {request.email}",
            status="FAILED",
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"reason": "no_otp_found"})
        )
        raise HTTPException(status_code=400, detail="No OTP found or already used")
    
    # Check if OTP is expired
    if is_otp_expired(otp_record.created_at, otp_record.expires_at):
        log_activity(
            db=db,
            activity_type="OTP_VERIFY_FAILED",
            description=f"OTP verification failed - OTP expired: {request.email}",
            status="FAILED",
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"reason": "otp_expired"})
        )
        raise HTTPException(status_code=400, detail="OTP has expired")
    
    # Verify OTP code
    if otp_record.otp_code != request.otp:
        log_activity(
            db=db,
            activity_type="OTP_VERIFY_FAILED",
            description=f"OTP verification failed - Invalid OTP: {request.email}",
            status="FAILED",
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"reason": "invalid_otp"})
        )
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    # Mark OTP as used
    otp_record.is_used = True
    
    # Activate user
    user = db.query(User).filter(User.email == request.email).first()
    if user:
        user.is_verified = True
        
        log_activity(
            db=db,
            activity_type="OTP_VERIFIED",
            description=f"OTP verified successfully: {request.email}",
            status="SUCCESS",
            user_id=user.id,
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"account_activated": True})
        )
    
    db.commit()
    
    return {
        "success": True,
        "message": "Email verified successfully. You can now login."
    }


@app.post("/api/auth/resend-otp")
async def resend_otp(request: ResendOTPRequest, req: Request, db: Session = Depends(get_db)):
    """Resend OTP to user"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_verified:
        raise HTTPException(status_code=400, detail="User already verified")
    
    # Generate new OTP
    otp_code = generate_otp()
    otp_expires = datetime.utcnow() + timedelta(minutes=10)
    
    new_otp = OTP(
        email=user.email,
        phone=user.phone,
        otp_code=otp_code,
        expires_at=otp_expires
    )
    
    db.add(new_otp)
    db.commit()
    
    # Send OTP
    send_otp_email(user.email, otp_code)
    send_otp_sms(user.phone, otp_code)
    
    log_activity(
        db=db,
        activity_type="OTP_RESEND",
        description=f"OTP resent to: {request.email}",
        status="SUCCESS",
        user_id=user.id,
        email=request.email,
        ip_address=client_ip,
        user_agent=user_agent,
        extra_data=json.dumps({"otp_sent": True})
    )
    
    return {
        "success": True,
        "message": "OTP resent successfully"
    }


@app.post("/api/auth/login")
async def login(request: LoginRequest, req: Request, db: Session = Depends(get_db)):
    """Login user and return access token"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    # Find user
    user = db.query(User).filter(User.email == request.email).first()
    
    if not user:
        log_activity(
            db=db,
            activity_type="LOGIN_FAILED",
            description=f"Login failed - User not found: {request.email}",
            status="FAILED",
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"reason": "user_not_found"})
        )
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Verify password
    if not verify_password(request.password, user.hashed_password):
        log_activity(
            db=db,
            activity_type="LOGIN_FAILED",
            description=f"Login failed - Invalid password: {request.email}",
            status="FAILED",
            user_id=user.id,
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"reason": "invalid_password"})
        )
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Check if verified
    if not user.is_verified:
        log_activity(
            db=db,
            activity_type="LOGIN_FAILED",
            description=f"Login failed - Email not verified: {request.email}",
            status="FAILED",
            user_id=user.id,
            email=request.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"reason": "email_not_verified"})
        )
        raise HTTPException(status_code=403, detail="Please verify your email first")
    
    # Create access token
    access_token = create_access_token(data={"sub": user.email})
    
    # Log successful login
    log_activity(
        db=db,
        activity_type="LOGIN",
        description=f"User logged in successfully: {request.email}",
        status="SUCCESS",
        user_id=user.id,
        email=request.email,
        ip_address=client_ip,
        user_agent=user_agent,
        extra_data=json.dumps({"token_issued": True, "is_admin": user.is_admin})
    )
    
    response = JSONResponse(content={
        "success": True,
        "message": "Login successful",
        "redirect": "/admin" if user.is_admin else "/dashboard",
        "is_admin": user.is_admin,
        "user": {
            "email": user.email,
            "phone": user.phone
        }
    })
    
    # Set cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=86400,  # 24 hours
        samesite="lax"
    )
    
    return response


@app.post("/api/auth/logout")
async def logout(req: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Logout user by clearing cookie"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    log_activity(
        db=db,
        activity_type="LOGOUT",
        description=f"User logged out: {current_user.email}",
        status="SUCCESS",
        user_id=current_user.id,
        email=current_user.email,
        ip_address=client_ip,
        user_agent=user_agent
    )
    
    response = JSONResponse(content={
        "success": True,
        "message": "Logged out successfully"
    })
    response.delete_cookie(key="access_token")
    return response


@app.get("/api/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "email": current_user.email,
        "phone": current_user.phone,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat()
    }


@app.post("/api/upload")
async def upload_file(req: Request, file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Upload CSV or XLSX file with advanced validation and processing (protected endpoint)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        log_activity(
            db=db,
            activity_type="FILE_UPLOAD",
            description=f"Upload failed: Invalid file type - {file.filename}",
            status="FAILED",
            user_id=current_user.id,
            email=current_user.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"filename": file.filename, "reason": "invalid_file_type"})
        )
        raise HTTPException(
            status_code=400, 
            detail="Only CSV and XLSX files are supported"
        )
    
    # Save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"dataset_{timestamp}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        content = await file.read()
        file_size = len(content)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Validate and process file with advanced loader
        try:
            df, quality_report = load_and_process_data(file_path)
            
            # Extract quality metrics
            duplicates_removed = quality_report['data_quality'].get('duplicates_removed', 0)
            missing_values_handled = len(quality_report['data_quality'].get('missing_values', {}))
            outliers_detected = len(quality_report['data_quality'].get('outliers', {}))
            
            log_activity(
                db=db,
                activity_type="FILE_UPLOAD",
                description=f"File uploaded and processed successfully: {file.filename}",
                status="SUCCESS",
                user_id=current_user.id,
                email=current_user.email,
                ip_address=client_ip,
                user_agent=user_agent,
                extra_data=json.dumps({
                    "original_filename": file.filename,
                    "saved_filename": safe_filename,
                    "file_size": file_size,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "data_quality": {
                        "duplicates_removed": duplicates_removed,
                        "missing_values_handled": missing_values_handled,
                        "outliers_detected": outliers_detected,
                        "encoding": quality_report['file_info'].get('encoding', 'N/A')
                    }
                })
            )
            
            return {
                "success": True,
                "filename": safe_filename,
                "filepath": file_path,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_quality": {
                    "duplicates_removed": duplicates_removed,
                    "missing_values_handled": missing_values_handled,
                    "outliers_detected": outliers_detected,
                    "encoding": quality_report['file_info'].get('encoding', 'UTF-8'),
                    "file_type": file_extension.upper()
                }
            }
            
        except Exception as e:
            # If advanced processing fails, try basic pandas read as fallback
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            log_activity(
                db=db,
                activity_type="FILE_UPLOAD",
                description=f"File uploaded (basic processing): {file.filename}",
                status="SUCCESS",
                user_id=current_user.id,
                email=current_user.email,
                ip_address=client_ip,
                user_agent=user_agent,
                extra_data=json.dumps({
                    "original_filename": file.filename,
                    "saved_filename": safe_filename,
                    "file_size": file_size,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "advanced_processing": False,
                    "warning": str(e)
                })
            )
            
            return {
                "success": True,
                "filename": safe_filename,
                "filepath": file_path,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_quality": {
                    "duplicates_removed": 0,
                    "missing_values_handled": 0,
                    "outliers_detected": 0,
                    "encoding": "Unknown",
                    "file_type": file_extension.upper(),
                    "note": "Basic processing used"
                }
            }
    
    except Exception as e:
        # Clean up file if error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        log_activity(
            db=db,
            activity_type="FILE_UPLOAD",
            description=f"Upload error: {str(e)}",
            status="ERROR",
            user_id=current_user.id,
            email=current_user.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"filename": file.filename, "error": str(e)})
        )
        
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@app.get("/api/analyze/{filename}")
async def analyze_file_stream(req: Request, filename: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Analyze uploaded file with streaming progress updates (protected endpoint)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        log_activity(
            db=db,
            activity_type="DATA_ANALYSIS",
            description=f"Analysis failed: File not found - {filename}",
            status="FAILED",
            user_id=current_user.id,
            email=current_user.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"filename": filename, "reason": "file_not_found"})
        )
        raise HTTPException(status_code=404, detail="File not found")
    
    # Log analysis start
    log_activity(
        db=db,
        activity_type="DATA_ANALYSIS",
        description=f"Analysis started: {filename}",
        status="STARTED",
        user_id=current_user.id,
        email=current_user.email,
        ip_address=client_ip,
        user_agent=user_agent,
        extra_data=json.dumps({"filename": filename})
    )
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events for progress updates"""
        try:
            async for progress_data in analyze_dataset_streaming(file_path):
                # Send as Server-Sent Event
                yield f"data: {json.dumps(progress_data)}\n\n"
                await asyncio.sleep(0.1)  # Small delay for smooth streaming
            
            # Log successful completion
            log_activity(
                db=db,
                activity_type="DATA_ANALYSIS",
                description=f"Analysis completed: {filename}",
                status="SUCCESS",
                user_id=current_user.id,
                email=current_user.email,
                ip_address=client_ip,
                user_agent=user_agent,
                extra_data=json.dumps({"filename": filename})
            )
        except Exception as e:
            # Log error
            log_activity(
                db=db,
                activity_type="DATA_ANALYSIS",
                description=f"Analysis error: {str(e)}",
                status="ERROR",
                user_id=current_user.id,
                email=current_user.email,
                ip_address=client_ip,
                user_agent=user_agent,
                extra_data=json.dumps({"filename": filename, "error": str(e)})
            )
            error_data = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete uploaded file"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"success": True, "message": "File deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.get("/api/logs")
async def get_activity_logs(
    req: Request,
    skip: int = 0,
    limit: int = 50,
    activity_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get activity logs (users can see their own logs)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        # Build query for user's own logs
        query = db.query(ActivityLog).filter(ActivityLog.user_id == current_user.id)
        
        # Filter by activity type if specified
        if activity_type:
            query = query.filter(ActivityLog.activity_type == activity_type)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        logs = query.order_by(ActivityLog.created_at.desc()).offset(skip).limit(limit).all()
        
        # Convert to dict
        logs_data = [
            {
                "id": log.id,
                "activity_type": log.activity_type,
                "description": log.description,
                "status": log.status,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "extra_data": log.extra_data,
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ]
        
        # Log the logs access
        log_activity(
            db=db,
            activity_type="LOGS_VIEW",
            description="User viewed activity logs",
            status="SUCCESS",
            user_id=current_user.id,
            email=current_user.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"skip": skip, "limit": limit, "activity_type": activity_type})
        )
        
        return {
            "success": True,
            "total": total,
            "skip": skip,
            "limit": limit,
            "logs": logs_data
        }
    
    except Exception as e:
        log_activity(
            db=db,
            activity_type="LOGS_VIEW",
            description=f"Error viewing logs: {str(e)}",
            status="ERROR",
            user_id=current_user.id,
            email=current_user.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"error": str(e)})
        )
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {str(e)}")


@app.get("/api/user/info")
async def get_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "success": True,
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "phone": current_user.phone,
            "is_verified": current_user.is_verified,
            "is_admin": current_user.is_admin,
            "created_at": current_user.created_at.isoformat()
        }
    }


# ==================== ADMIN ENDPOINTS ====================

@app.get("/api/admin/users")
async def get_all_users(
    req: Request,
    skip: int = 0,
    limit: int = 50,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        # Get total count
        total = db.query(User).count()
        
        # Get paginated users
        users = db.query(User).order_by(User.created_at.desc()).offset(skip).limit(limit).all()
        
        users_data = [
            {
                "id": user.id,
                "email": user.email,
                "phone": user.phone,
                "is_verified": user.is_verified,
                "is_admin": user.is_admin,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat()
            }
            for user in users
        ]
        
        log_activity(
            db=db,
            activity_type="ADMIN_VIEW_USERS",
            description="Admin viewed users list",
            status="SUCCESS",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"total_users": total})
        )
        
        return {
            "success": True,
            "total": total,
            "skip": skip,
            "limit": limit,
            "users": users_data
        }
    except Exception as e:
        log_activity(
            db=db,
            activity_type="ADMIN_VIEW_USERS",
            description=f"Error viewing users: {str(e)}",
            status="ERROR",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"error": str(e)})
        )
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")


@app.get("/api/admin/logs")
async def get_all_activity_logs(
    req: Request,
    skip: int = 0,
    limit: int = 50,
    activity_type: Optional[str] = None,
    user_id: Optional[int] = None,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get all activity logs with filters (admin only)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        # Build query
        query = db.query(ActivityLog)
        
        # Filter by activity type if specified
        if activity_type:
            query = query.filter(ActivityLog.activity_type == activity_type)
        
        # Filter by user_id if specified
        if user_id:
            query = query.filter(ActivityLog.user_id == user_id)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        logs = query.order_by(ActivityLog.created_at.desc()).offset(skip).limit(limit).all()
        
        # Convert to dict
        logs_data = [
            {
                "id": log.id,
                "user_id": log.user_id,
                "email": log.email,
                "activity_type": log.activity_type,
                "description": log.description,
                "status": log.status,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "extra_data": log.extra_data,
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ]
        
        log_activity(
            db=db,
            activity_type="ADMIN_VIEW_LOGS",
            description="Admin viewed activity logs",
            status="SUCCESS",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"skip": skip, "limit": limit, "activity_type": activity_type, "user_id": user_id})
        )
        
        return {
            "success": True,
            "total": total,
            "skip": skip,
            "limit": limit,
            "logs": logs_data
        }
    
    except Exception as e:
        log_activity(
            db=db,
            activity_type="ADMIN_VIEW_LOGS",
            description=f"Error viewing logs: {str(e)}",
            status="ERROR",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"error": str(e)})
        )
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {str(e)}")


@app.get("/api/admin/stats")
async def get_admin_stats(
    req: Request,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get platform statistics (admin only)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        total_users = db.query(User).count()
        verified_users = db.query(User).filter(User.is_verified == True).count()
        admin_users = db.query(User).filter(User.is_admin == True).count()
        total_logs = db.query(ActivityLog).count()
        
        # Recent activity (last 24 hours)
        last_24h = datetime.utcnow() - timedelta(hours=24)
        recent_signups = db.query(User).filter(User.created_at >= last_24h).count()
        recent_logins = db.query(ActivityLog).filter(
            ActivityLog.activity_type == "LOGIN",
            ActivityLog.created_at >= last_24h
        ).count()
        
        # Activity by type
        from sqlalchemy import func
        activity_counts = db.query(
            ActivityLog.activity_type,
            func.count(ActivityLog.id).label('count')
        ).group_by(ActivityLog.activity_type).all()
        
        activity_by_type = {activity: count for activity, count in activity_counts}
        
        log_activity(
            db=db,
            activity_type="ADMIN_VIEW_STATS",
            description="Admin viewed platform statistics",
            status="SUCCESS",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        return {
            "success": True,
            "stats": {
                "total_users": total_users,
                "verified_users": verified_users,
                "admin_users": admin_users,
                "total_logs": total_logs,
                "recent_signups_24h": recent_signups,
                "recent_logins_24h": recent_logins,
                "activity_by_type": activity_by_type
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


class UpdateUserRequest(BaseModel):
    is_verified: Optional[bool] = None
    is_admin: Optional[bool] = None


@app.put("/api/admin/users/{user_id}")
async def update_user(
    req: Request,
    user_id: int,
    update_data: UpdateUserRequest,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Update user (admin only)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        updated_fields = {}
        
        if update_data.is_verified is not None:
            user.is_verified = update_data.is_verified
            updated_fields["is_verified"] = update_data.is_verified
        
        if update_data.is_admin is not None:
            user.is_admin = update_data.is_admin
            updated_fields["is_admin"] = update_data.is_admin
        
        user.updated_at = datetime.utcnow()
        db.commit()
        
        log_activity(
            db=db,
            activity_type="ADMIN_UPDATE_USER",
            description=f"Admin updated user {user.email}",
            status="SUCCESS",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"target_user_id": user_id, "updates": updated_fields})
        )
        
        return {
            "success": True,
            "message": "User updated successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "is_verified": user.is_verified,
                "is_admin": user.is_admin
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        log_activity(
            db=db,
            activity_type="ADMIN_UPDATE_USER",
            description=f"Error updating user: {str(e)}",
            status="ERROR",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"target_user_id": user_id, "error": str(e)})
        )
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")


@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    req: Request,
    user_id: int,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Don't allow deleting yourself
        if user.id == current_admin.id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")
        
        user_email = user.email
        db.delete(user)
        db.commit()
        
        log_activity(
            db=db,
            activity_type="ADMIN_DELETE_USER",
            description=f"Admin deleted user {user_email}",
            status="SUCCESS",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"deleted_user_id": user_id, "deleted_email": user_email})
        )
        
        return {
            "success": True,
            "message": "User deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        log_activity(
            db=db,
            activity_type="ADMIN_DELETE_USER",
            description=f"Error deleting user: {str(e)}",
            status="ERROR",
            user_id=current_admin.id,
            email=current_admin.email,
            ip_address=client_ip,
            user_agent=user_agent,
            extra_data=json.dumps({"target_user_id": user_id, "error": str(e)})
        )
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


