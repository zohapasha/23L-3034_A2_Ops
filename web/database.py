import os
import time
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(255))
    size_kb: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    p_ai: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(100))
    mode: Mapped[str] = mapped_column(String(10))
    latency_ms: Mapped[int] = mapped_column(Integer)


engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///visionguard.db"), pool_pre_ping=True)


def init_db(tries=10):
    # postgres can take a few seconds to start in docker
    for i in range(1, tries + 1):
        try:
            Base.metadata.create_all(engine)
            print("connected to", engine.url.render_as_string(hide_password=True), flush=True)
            return
        except Exception as e:
            print(f"db connection failed ({i}/{tries}): {e}", flush=True)
            if i == tries:
                raise
            time.sleep(3)


def save(**fields):
    with Session(engine) as s:
        row = Prediction(**fields)
        s.add(row)
        s.commit()
        return row.id


def latest(n=100):
    with Session(engine) as s:
        return list(s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(n)))


def is_up():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
