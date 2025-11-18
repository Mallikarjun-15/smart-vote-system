# SmartVote – Face-Verified Online Voting

SmartVote is a Streamlit-based election platform where registered voters cast a single ballot only after passing biometric face verification and liveness checks. Admins can create elections, add candidates, and monitor live results while the system prevents duplicate registrations, double voting, and impersonation.

## Features
- Streamlit UI with password-based login and session state
- Face enrollment and verification powered by DeepFace embeddings
- Basic liveness detection via Laplacian variance plus optional DeepFace anti-spoofing
- SQLite database with SQLAlchemy models for users, elections, candidates, and votes
- Admin console for managing elections/candidates and viewing results
- Rate limiting of failed biometric attempts

## Project Structure
```
smartvote/
├─ app.py               # Streamlit app
├─ db.py                # SQLAlchemy models and session helpers
├─ auth.py              # Password hashing + auth utilities
├─ admin.py             # Admin-facing election logic
├─ face_utils.py        # DeepFace embeddings, verification, liveness
├─ requirements.txt     # Python dependencies
├─ images/              # Saved ID and face captures
└─ README.md
```

## Prerequisites
- Python 3.11+ (tested on Windows with Python 3.13)
- pip / virtual environment recommended
- Webcam access in the browser to capture enrollment/voting images

## Setup
```bash
git clone https://github.com/Mallikarjun-15/smart-vote-system.git
cd smart-vote-system
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

Optional environment variables:
```
SMARTVOTE_DB_URL=sqlite:///smartvote.db   # default
SMARTVOTE_ADMIN_CODE=ADMIN123             # invite code during registration
```

## Running Locally
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (default http://localhost:8501).

### Workflow
1. **Register** – provide name, email, password, optional phone & ID, and capture a face image.
2. **Login** – authenticate with email/password.
3. **Vote** – choose an active election, capture a live image, and vote via candidate buttons.
4. **Admin Console** – register with the invite code or use the seeded admin account:
   - Email: `malluratkal@gmail.com`
   - Password: `123456`
   - Create elections, add candidates, and view vote counts.

## Deployment (Streamlit Cloud)
1. Push the repository to GitHub.
2. Visit https://share.streamlit.io/, connect your GitHub account, and deploy `app.py`.
3. Configure any secrets/environment variables in Streamlit’s settings.

## Notes
- DeepFace downloads required weights on first run; allow time for initial load.
- SQLite stores embeddings as blobs; raw face images remain locally under `images/`.
- For production, consider moving to PostgreSQL, enabling HTTPS, and adding MFA/OTP.

## License
This project is provided as-is for demonstration purposes. Customize security controls before running in production environments.

