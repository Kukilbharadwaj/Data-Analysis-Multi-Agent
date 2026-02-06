# AI Data Analyst Platform

An intelligent multi-agent system for automated data analysis with user authentication and admin dashboard.

## Summary

This platform uses 6 specialized AI agents (Planner, Explorer, SQL, Insight, Validator, Narrator) powered by LangGraph and Groq LLM to collaboratively analyze datasets. Features include user authentication, admin management, real-time analysis tracking, and support for CSV/XLSX files.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python backend.py
```

Access at: **http://localhost:8000**

## Features

- **Multi-Agent Analysis** - 6 AI agents working collaboratively
- **User Authentication** - Secure login/signup with email verification
- **Admin Dashboard** - User management and system monitoring
- **Real-time Updates** - Live progress tracking via Server-Sent Events
- **File Support** - CSV and XLSX file upload and analysis
- **Docker Ready** - Container deployment included

## Tech Stack

- **Backend**: FastAPI, SQLite, SQLAlchemy
- **AI**: LangGraph, Groq (Llama 3.3), LangChain
- **Frontend**: HTML, JavaScript, Tailwind CSS
- **Data**: Pandas, OpenPyXL

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | User registration |
| POST | `/auth/login` | User login |
| POST | `/api/upload` | Upload dataset |
| GET | `/api/analyze/{filename}` | Run analysis |
| GET | `/admin/users` | Admin user management |

## Admin Setup

```bash
python make_admin.py <email>
```

## License

MIT
- Time series data (date, value, category)

---
