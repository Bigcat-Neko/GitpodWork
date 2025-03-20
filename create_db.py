# create_db.py
from app.database import Base, engine

# Create all tables defined in your models
Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")
