from app import db, app
with app.app_context():
    db.create_all()
    print("database.db created (tables ensured).")