import sqlite3
import os
import io
import numpy as np

class DatabaseManager:
    def __init__(self, db_path='face_db.sqlite'):
        self.db_path = db_path
        
        # Register numpy adapter and converter to easily save/load embeddings
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        
        self.init_db()

    def adapt_array(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def init_db(self):
        conn = self.get_connection()
        c = conn.cursor()
        # Create table for identities
        c.execute('''
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('VIP', 'Blacklist')),
                image_path TEXT,
                embedding array
            )
        ''')
        conn.commit()
        conn.close()

    def get_connection(self):
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def add_identity(self, name, category, image_path, embedding):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO identities (name, category, image_path, embedding)
            VALUES (?, ?, ?, ?)
        ''', (name, category, image_path, embedding))
        conn.commit()
        conn.close()

    def get_all_identities(self):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('SELECT id, name, category, image_path, embedding FROM identities')
        rows = c.fetchall()
        conn.close()
        
        identities = []
        for row in rows:
            identities.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'image_path': row[3],
                'embedding': row[4]
            })
        return identities

    def delete_identity(self, identity_id):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('DELETE FROM identities WHERE id = ?', (identity_id,))
        conn.commit()
        conn.close()
    
    def get_identities_by_category(self, category):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('SELECT id, name, category, image_path FROM identities WHERE category = ?', (category,))
        rows = c.fetchall()
        conn.close()
        
        identities = []
        for row in rows:
            identities.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'image_path': row[3],
            })
        return identities
