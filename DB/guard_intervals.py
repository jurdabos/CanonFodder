"""
Adds guards that prevent overlapping date ranges in `UserCountry` rows.
"""
from datetime import date, timedelta
from sqlalchemy import event, func
from sqlalchemy.orm import Session

# Helper constant keeps an “infinite” upper-bound date
DATE_MAX = date.max - timedelta(days=1)  # 9999-12-30


def _overlap_clause(cls, new_start, new_end):
    """
    Builds an SQLAlchemy filter that matches any row overlapping the
    half-open interval [`new_start`, `new_end`). `new_end` may be None
    """
    new_end = new_end or DATE_MAX
    return (
        cls.start_date < new_end,
        func.coalesce(cls.end_date, DATE_MAX) > new_start,
    )


@event.listens_for(Session, "before_flush")
def no_overlapping_user_countries(session: Session, flush_ctx, _):
    """
    Aborts the flush if a new or updated `UserCountry` row overlaps an
    existing period for the same `country_name`
    """
    from DB.models import UserCountry  # local import to avoid cycles
    for obj in session.new.union(session.dirty):
        if not isinstance(obj, UserCountry):
            continue
        # Ignoring rows being deleted
        if session.is_deleted(obj):
            continue
        # Does *another* row overlap?
        overlap_q = (
            session.query(UserCountry)
            .filter(UserCountry.id != getattr(obj, "id", None))  # different row
            .filter(UserCountry.country_name == obj.country_name)  # same country
            .filter(*_overlap_clause(UserCountry, obj.start_date, obj.end_date))
        )
        if session.query(overlap_q.exists()).scalar():
            raise ValueError(
                f"Overlap detected for {obj.country_name}: "
                f"{obj.start_date} – {obj.end_date or '∞'}"
            )
