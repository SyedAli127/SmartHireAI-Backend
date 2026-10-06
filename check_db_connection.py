from sqlalchemy import text

from database.session import get_engine


def check_db_connection() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connected successfully.")
        return True
    except Exception as exc:
        print(f"Database connection failed: {exc}")
        return False


if __name__ == "__main__":
    check_db_connection()
