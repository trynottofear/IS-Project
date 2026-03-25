import sqlite3
import os

db_path = 'face_db.sqlite'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys = OFF')
    c = conn.cursor()
    c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='embeddings'")
    res = c.fetchone()
    if res and 'old_identities' in res[0]:
        print('Fixing database foreign keys...')
        c.execute('ALTER TABLE embeddings RENAME TO old_embeddings_broken_fk')
        c.execute('''
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_id INTEGER NOT NULL,
                image_path TEXT,
                embedding array,
                FOREIGN KEY(identity_id) REFERENCES identities(id) ON DELETE CASCADE
            )
        ''')
        c.execute('SELECT id, identity_id, image_path, embedding FROM old_embeddings_broken_fk')
        rows = c.fetchall()
        for row in rows:
            c.execute('INSERT INTO embeddings (id, identity_id, image_path, embedding) VALUES (?, ?, ?, ?)', row)
        c.execute('DROP TABLE old_embeddings_broken_fk')
        conn.commit()
        print('Database Fixed!')
    else:
        print('DB already correct or no fix needed.')
    conn.close()
