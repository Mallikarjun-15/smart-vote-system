from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import func

from db import SessionLocal, Candidate, Election, Vote


def create_election(
    session: SessionLocal,
    *,
    title: str,
    description: str,
    start_time: datetime,
    end_time: datetime,
) -> Election:
    election = Election(
        title=title.strip(),
        description=description.strip(),
        start_time=start_time,
        end_time=end_time,
    )
    session.add(election)
    session.commit()
    session.refresh(election)
    return election


def add_candidate(session: SessionLocal, election_id: int, name: str, party: str = "") -> Candidate:
    candidate = Candidate(election_id=election_id, name=name.strip(), party=party.strip())
    session.add(candidate)
    session.commit()
    session.refresh(candidate)
    return candidate


def list_active_elections(session: SessionLocal, include_future: bool = False) -> List[Election]:
    now = datetime.utcnow()
    query = session.query(Election)
    if include_future:
        query = query.filter(Election.end_time >= now)
    else:
        query = query.filter(Election.start_time <= now, Election.end_time >= now)
    return query.order_by(Election.start_time).all()


def list_all_elections(session: SessionLocal) -> List[Election]:
    return session.query(Election).order_by(Election.start_time.desc()).all()


def has_user_voted(session: SessionLocal, voter_id: int, election_id: int) -> bool:
    return (
        session.query(Vote)
        .filter(Vote.voter_id == voter_id, Vote.election_id == election_id)
        .first()
        is not None
    )


def record_vote(
    session: SessionLocal,
    *,
    voter_id: int,
    election_id: int,
    candidate_id: int,
) -> Vote:
    if has_user_voted(session, voter_id, election_id):
        raise ValueError("User has already voted in this election.")
    vote = Vote(voter_id=voter_id, election_id=election_id, candidate_id=candidate_id)
    session.add(vote)
    session.commit()
    session.refresh(vote)
    return vote


def election_results(session: SessionLocal, election_id: int) -> List[Tuple[Candidate, int]]:
    results = (
        session.query(Candidate, func.count(Vote.id).label("vote_count"))
        .outerjoin(Vote, Vote.candidate_id == Candidate.id)
        .filter(Candidate.election_id == election_id)
        .group_by(Candidate.id)
        .order_by(func.count(Vote.id).desc())
        .all()
    )
    return results


def get_election(session: SessionLocal, election_id: int) -> Optional[Election]:
    return session.query(Election).filter(Election.id == election_id).first()

