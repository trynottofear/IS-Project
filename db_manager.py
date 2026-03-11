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
        
        # Check if old table with 'embedding' column exists
        c.execute("PRAGMA table_info(identities)")
        columns = [info[1] for info in c.fetchall()]
        
        if 'embedding' in columns:
            self._migrate_db(conn)
        else:
            # Create new tables
            c.execute('''
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL CHECK(category IN ('VIP', 'Blacklist'))
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id INTEGER NOT NULL,
                    image_path TEXT,
                    embedding array,
                    FOREIGN KEY(identity_id) REFERENCES identities(id) ON DELETE CASCADE
                )
            ''')
        
        conn.commit()
        conn.close()

    def _migrate_db(self, conn):
        c = conn.cursor()
        print("Migrating database to V2 schema...")
        
        # Temporarily rename old table
        c.execute("ALTER TABLE identities RENAME TO old_identities")
        
        # Create new tables
        c.execute('''
            CREATE TABLE identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('VIP', 'Blacklist'))
            )
        ''')
        
        c.execute('''
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_id INTEGER NOT NULL,
                image_path TEXT,
                embedding array,
                FOREIGN KEY(identity_id) REFERENCES identities(id) ON DELETE CASCADE
            )
        ''')
        
        # Move data
        c.execute("SELECT id, name, category, image_path, embedding FROM old_identities")
        rows = c.fetchall()
        
        for row in rows:
            old_id, name, category, image_path, embedding = row
            # Insert identity
            c.execute("INSERT INTO identities (name, category) VALUES (?, ?)", (name, category))
            new_id = c.lastrowid
            
            # Insert its single embedding
            c.execute("INSERT INTO embeddings (identity_id, image_path, embedding) VALUES (?, ?, ?)", 
                      (new_id, image_path, embedding))
                      
        # Drop old table
        c.execute("DROP TABLE old_identities")
        print("Migration complete!")

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = 1")
        return conn

    def add_identity(self, name, category, images_data):
        """
        images_data should be a list of dicts: [{'path': str, 'embedding': array}, ...]
        """
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('INSERT INTO identities (name, category) VALUES (?, ?)', (name, category))
        identity_id = c.lastrowid
        
        for item in images_data:
            c.execute('''
                INSERT INTO embeddings (identity_id, image_path, embedding)
                VALUES (?, ?, ?)
            ''', (identity_id, item['path'], item['embedding']))
            
        conn.commit()
        conn.close()
        return identity_id
        
    def add_embedding(self, identity_id, image_path, embedding):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO embeddings (identity_id, image_path, embedding)
            VALUES (?, ?, ?)
        ''', (identity_id, image_path, embedding))
        conn.commit()
        conn.close()

    def get_all_identities_with_embeddings(self):
        conn = self.get_connection()
        c = conn.cursor()
        
        # Get all identities
        c.execute('SELECT id, name, category FROM identities')
        identities_rows = c.fetchall()
        
        identities = []
        for id_row in identities_rows:
            identity_id = id_row[0]
            
            # Get all embeddings for this identity
            c.execute('SELECT id, image_path, embedding FROM embeddings WHERE identity_id = ?', (identity_id,))
            emb_rows = c.fetchall()
            
            embeddings = []
            for e_row in emb_rows:
                embeddings.append({
                    'id': e_row[0],
                    'image_path': e_row[1],
                    'embedding': e_row[2]
                })
                
            identities.append({
                'id': identity_id,
                'name': id_row[1],
                'category': id_row[2],
                'embeddings': embeddings
            })
            
        conn.close()
        return identities

    def update_identity(self, identity_id, name, category):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('UPDATE identities SET name = ?, category = ? WHERE id = ?', (name, category, identity_id))
        conn.commit()
        conn.close()

    def delete_identity(self, identity_id):
        conn = self.get_connection()
        c = conn.cursor()
        # ON DELETE CASCADE handles deleting the embeddings
        c.execute('DELETE FROM identities WHERE id = ?', (identity_id,))
        conn.commit()
        conn.close()
    
    def delete_embedding(self, embedding_id):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('DELETE FROM embeddings WHERE id = ?', (embedding_id,))
        conn.commit()
        conn.close()
