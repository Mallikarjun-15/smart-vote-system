import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from admin import (
    add_candidate,
    create_election,
    election_results,
    has_user_voted,
    list_active_elections,
    list_all_elections,
    record_vote,
)
from auth import (
    authenticate_user,
    can_attempt_face_verification,
    get_user_by_id,
    record_failed_face_attempt,
    register_user,
    reset_failed_face_attempts,
)
from db import Candidate, Election, SessionLocal, Vote, init_db, session_scope
from face_utils import (
    blob_to_embedding,
    deepface_liveness_check,
    embedding_to_blob,
    generate_face_embedding,
    perform_basic_liveness_check,
    save_image_bytes,
    verify_face,
)


IMAGES_ROOT = Path("images")
FACE_DIR = IMAGES_ROOT / "faces"
ID_DIR = IMAGES_ROOT / "ids"
ADMIN_INVITE_CODE = os.getenv("SMARTVOTE_ADMIN_CODE", "ADMIN123")
LIVENESS_VARIANCE_THRESHOLD = 6.0


@st.cache_resource
def _bootstrap():
    IMAGES_ROOT.mkdir(exist_ok=True)
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    ID_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    return True


_ = _bootstrap()

st.set_page_config(page_title="SmartVote - Secure Face Verified Voting", layout="wide")


def get_db_session():
    return SessionLocal()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def snapshot_current_user() -> Optional[Dict]:
    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        return None
    with session_scope() as session:
        user = get_user_by_id(session, user_id)
        if not user:
            st.session_state.clear()
            return None
        return {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role,
            "created_at": user.created_at,
        }


def logout():
    st.session_state.pop("auth_user_id", None)
    st.session_state.pop("auth_role", None)
    st.session_state.pop("auth_name", None)
    st.success("Logged out successfully.")


def login_user(user):
    st.session_state["auth_user_id"] = user.id
    st.session_state["auth_role"] = user.role
    st.session_state["auth_name"] = user.full_name
    st.success(f"Welcome back, {user.full_name}!")


def render_login():
    st.subheader("Login")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email").strip()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
    if submitted:
        if not email or not password:
            st.warning("Please provide both email and password.")
            return
        session = get_db_session()
        try:
            user = authenticate_user(session, email, password)
            if not user:
                st.error("Invalid credentials.")
            else:
                login_user(user)
        finally:
            session.close()


def render_registration():
    st.subheader("Register New Voter")
    st.caption("Face enrollment is mandatory and liveness-safe.")
    with st.form("registration_form", clear_on_submit=False):
        full_name = st.text_input("Full Name *")
        email = st.text_input("Email *")
        phone = st.text_input("Phone (optional)")
        password = st.text_input("Password *", type="password", help="Use a strong password.")
        gov_id = st.file_uploader("Government ID (image, optional)", type=["png", "jpg", "jpeg"])
        face_capture = st.camera_input("Capture Face Enrollment *")
        wants_admin = st.checkbox("Register as Admin (requires invite code)")
        admin_code = ""
        if wants_admin:
            admin_code = st.text_input("Admin Invite Code", type="password")
        submitted = st.form_submit_button("Create Account")

    if submitted:
        if not full_name or not email or not password:
            st.error("Please fill in all required fields.")
            return
        if not face_capture:
            st.error("Face enrollment capture is required.")
            return
        face_bytes = face_capture.getvalue()
        embedding = generate_face_embedding(face_bytes)
        if embedding is None:
            st.error("Face not detected. Please recapture with clear lighting.")
            return
        face_blob = embedding_to_blob(embedding)

        gov_id_path = None
        timestamp = int(datetime.utcnow().timestamp())
        slug = slugify(email)
        if gov_id is not None:
            gov_bytes = gov_id.getvalue()
            gov_id_path = ID_DIR / f"{slug}_{timestamp}_id.png"
            save_image_bytes(gov_bytes, gov_id_path)

        face_path = FACE_DIR / f"{slug}_{timestamp}_face.png"
        save_image_bytes(face_bytes, face_path)

        role = "admin" if wants_admin and admin_code == ADMIN_INVITE_CODE else "voter"
        if wants_admin and role != "admin":
            st.warning("Invalid admin invite code. Registering as voter.")

        session = get_db_session()
        try:
            register_user(
                session,
                full_name=full_name,
                email=email,
                password=password,
                phone=phone,
                role=role,
                face_embedding_blob=face_blob,
                gov_id_path=str(gov_id_path) if gov_id_path else None,
            )
            st.success("Registration successful. You may now log in.")
        except ValueError as exc:
            st.error(str(exc))
        finally:
            session.close()


def render_admin_panel(user_snapshot: Dict):
    st.subheader("Admin Console")
    admin_tabs = st.tabs(["Create Election", "Add Candidates", "Results"])

    with admin_tabs[0]:
        st.markdown("### Create a New Election")
        with st.form("create_election_form"):
            title = st.text_input("Election Title *")
            description = st.text_area("Description", height=100)
            now = datetime.utcnow()
            start_date = st.date_input("Start Date *", value=now.date())
            start_time = st.time_input("Start Time *", value=now.time())
            end_default = now + timedelta(days=7)
            end_date = st.date_input("End Date *", value=end_default.date(), key="end_date_input")
            end_time = st.time_input("End Time *", value=end_default.time(), key="end_time_input")
            submitted = st.form_submit_button("Create Election")
        if submitted:
            start = datetime.combine(start_date, start_time)
            end = datetime.combine(end_date, end_time)
            if not title:
                st.error("Please supply required fields.")
            elif end <= start:
                st.error("End time must be after start time.")
            else:
                with session_scope() as session:
                    create_election(
                        session,
                        title=title,
                        description=description,
                        start_time=start,
                        end_time=end,
                    )
                st.success("Election created.")

    with admin_tabs[1]:
        st.markdown("### Add Candidates to Election")
        with session_scope() as session:
            elections = list_all_elections(session)
        if not elections:
            st.info("No elections available. Create one first.")
        else:
            election_map = {f"{e.title} ({e.id})": e.id for e in elections}
            with st.form("add_candidate_form"):
                election_label = st.selectbox("Select Election *", list(election_map.keys()))
                candidate_name = st.text_input("Candidate Name *")
                candidate_party = st.text_input("Party / Affiliation")
                submitted = st.form_submit_button("Add Candidate")
            if submitted:
                if not candidate_name:
                    st.error("Candidate name is required.")
                else:
                    with session_scope() as session:
                        add_candidate(
                            session,
                            election_id=election_map[election_label],
                            name=candidate_name,
                            party=candidate_party,
                        )
                    st.success("Candidate added.")

    with admin_tabs[2]:
        st.markdown("### Election Results")
        with session_scope() as session:
            elections = list_all_elections(session)
        if not elections:
            st.info("No elections yet.")
        else:
            election_choice = st.selectbox(
                "Select election to view results",
                options=elections,
                format_func=lambda e: f"{e.title} ({e.start_time.date()} - {e.end_time.date()})",
            )
            if election_choice:
                with session_scope() as session:
                    results = election_results(session, election_choice.id)
                st.write(f"**{election_choice.title}** Results")
                for candidate, count in results:
                    st.metric(label=f"{candidate.name} ({candidate.party or 'Independent'})", value=count)


def render_voting_section(user_snapshot: Dict):
    st.subheader("Available Elections")
    with session_scope() as session:
        elections = list_active_elections(session, include_future=False)
        election_payload = []
        for election in elections:
            candidates = [
                {"id": candidate.id, "name": candidate.name, "party": candidate.party}
                for candidate in election.candidates
            ]
            election_payload.append(
                {
                    "id": election.id,
                    "title": election.title,
                    "description": election.description,
                    "start_time": election.start_time,
                    "end_time": election.end_time,
                    "candidates": candidates,
                }
            )

    if not election_payload:
        st.info("No active elections right now.")
        return

    for election in election_payload:
        st.markdown(f"### {election['title']}")
        st.caption(
            f"{election['description'] or ''}\n"
            f"Open from {election['start_time']} to {election['end_time']}"
        )
        with session_scope() as session:
            already_voted = has_user_voted(session, user_snapshot["id"], election["id"])
        if already_voted:
            st.success("You have already voted in this election.")
            continue

        if not election["candidates"]:
            st.warning("No candidates added yet.")
            continue

        live_capture = st.camera_input(
            "Capture live image for verification",
            key=f"cam_{election['id']}",
        )
        enable_antispoof = st.checkbox(
            "Use DeepFace anti-spoofing (slower)",
            key=f"anti_{election['id']}",
        )

        st.markdown("**Choose your candidate:**")
        cols = st.columns(len(election["candidates"]))
        for idx, candidate in enumerate(election["candidates"]):
            with cols[idx]:
                st.markdown(
                    f"**{candidate['name']}**  \n"
                    f"{candidate['party'] or 'Independent'}"
                )
                if st.button(
                    f"Vote for {candidate['name']}",
                    key=f"vote_btn_{election['id']}_{candidate['id']}",
                    use_container_width=True,
                ):
                    process_vote_submission(
                        user_snapshot=user_snapshot,
                        election_id=election["id"],
                        candidate_id=candidate["id"],
                        live_capture=live_capture,
                        enable_antispoof=enable_antispoof,
                    )


def process_vote_submission(
    *,
    user_snapshot: Dict,
    election_id: int,
    candidate_id: int,
    live_capture,
    enable_antispoof: bool,
):
    if not live_capture:
        st.error("Live capture required for face verification.")
        return

    live_bytes = live_capture.getvalue()

    with session_scope() as session:
        user = get_user_by_id(session, user_snapshot["id"])
        if not user or not user.face_embedding:
            st.error("Face enrollment missing. Contact administrator.")
            return
        if not can_attempt_face_verification(user):
            st.error("Too many failed attempts. Please try again later.")
            return
        registered_embedding = blob_to_embedding(user.face_embedding)

    liveness_ok, variance = perform_basic_liveness_check(live_bytes, LIVENESS_VARIANCE_THRESHOLD)
    if not liveness_ok:
        st.error(
            f"Liveness verification failed (variance={variance:.2f}). "
            "Ensure real-time capture with good lighting."
        )
        with session_scope() as session:
            user = get_user_by_id(session, user_snapshot["id"])
            record_failed_face_attempt(session, user)
        return

    if enable_antispoof:
        spoof_ok, spoof_msg = deepface_liveness_check(live_bytes)
        if not spoof_ok:
            st.error(f"DeepFace anti-spoofing failed: {spoof_msg}")
            with session_scope() as session:
                user = get_user_by_id(session, user_snapshot["id"])
                record_failed_face_attempt(session, user)
            return
        st.info(spoof_msg)

    live_embedding = generate_face_embedding(live_bytes)
    if live_embedding is None:
        st.error("Could not detect face in live capture. Please try again.")
        return

    match, distance = verify_face(registered_embedding, live_embedding)
    if not match:
        st.error(f"Face mismatch detected (distance={distance:.2f}).")
        with session_scope() as session:
            user = get_user_by_id(session, user_snapshot["id"])
            record_failed_face_attempt(session, user)
        return

    with session_scope() as session:
        user = get_user_by_id(session, user_snapshot["id"])
        if has_user_voted(session, user.id, election_id):
            st.warning("You have already voted in this election.")
            return
        record_vote(
            session,
            voter_id=user.id,
            election_id=election_id,
            candidate_id=candidate_id,
        )
        reset_failed_face_attempts(session, user)
    st.success("Vote recorded successfully with verified face match.")


def render_my_votes(user_snapshot: Dict):
    st.subheader("My Voting History")
    with session_scope() as session:
        rows = (
            session.query(Vote, Election, Candidate)
            .join(Election, Vote.election_id == Election.id)
            .join(Candidate, Vote.candidate_id == Candidate.id)
            .filter(Vote.voter_id == user_snapshot["id"])
            .order_by(Vote.timestamp.desc())
            .all()
        )
    history = [
        {
            "election_title": election.title,
            "candidate_name": candidate.name,
            "candidate_party": candidate.party,
            "timestamp": vote.timestamp,
        }
        for vote, election, candidate in rows
    ]

    if not history:
        st.info("No votes recorded yet.")
        return
    for entry in history:
        st.write(
            f"- **{entry['election_title']}** → {entry['candidate_name']} "
            f"({entry['candidate_party'] or 'Independent'}) "
            f"on {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )


def render_dashboard():
    user_snapshot = snapshot_current_user()
    if not user_snapshot:
        st.info("Please log in to access the dashboard.")
        return

    st.success(f"Logged in as {user_snapshot['full_name']} ({user_snapshot['role']})")
    if st.button("Logout", use_container_width=False):
        logout()
        st.stop()

    voting_tab, history_tab = st.tabs(["Voting", "My Votes"])

    with voting_tab:
        render_voting_section(user_snapshot)
    with history_tab:
        render_my_votes(user_snapshot)

    if user_snapshot["role"] == "admin":
        st.divider()
        render_admin_panel(user_snapshot)


def main():
    st.title("SmartVote - Face Verified Voting System")
    st.write(
        "Secure single-vote elections with biometric face verification, liveness checks, "
        "and admin-controlled election management."
    )

    if "auth_user_id" in st.session_state:
        render_dashboard()
    else:
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            render_login()
        with register_tab:
            render_registration()


if __name__ == "__main__":
    main()

