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
        fetchall = c.fetchall()
        columns = [info[1] for info in fetchall] if fetchall else []
        
        # Check schema text for 'Unknown' constraint
        c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='identities'")
        res = c.fetchone()
        sql_text = res[0] if res else ""
        
        if 'embedding' in columns:
            self._migrate_db(conn)
            if 'Unknown' not in sql_text:
                self._migrate_db_v3(conn)
        elif sql_text and 'Unknown' not in sql_text:
            self._migrate_db_v3(conn)
        else:
            # Create new tables
            c.execute('''
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL CHECK(category IN ('VIP', 'Blacklist', 'Unknown'))
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
            
        # Self-healing check for broken foreign keys caused by SQLite ALTER TABLE RENAME
        c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='embeddings'")
        res = c.fetchone()
        emb_sql = res[0] if res else ""
        if 'old_identities' in emb_sql:
            self._fix_foreign_keys(conn)
            
        c.execute('''
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_id INTEGER,
                name TEXT,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
                category TEXT NOT NULL CHECK(category IN ('VIP', 'Blacklist', 'Unknown'))
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

    def _migrate_db_v3(self, conn):
        c = conn.cursor()
        print("Migrating database to V3 schema (adding 'Unknown' category)...")
        
        # Temporarily rename old table
        c.execute("ALTER TABLE identities RENAME TO old_identities_v2")
        
        # Create new tables
        c.execute('''
            CREATE TABLE identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('VIP', 'Blacklist', 'Unknown'))
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
        
        # Move data
        c.execute("SELECT id, name, category FROM old_identities_v2")
        rows = c.fetchall()
        
        for row in rows:
            old_id, name, category = row
            # We want to preserve the old ID so the foreign keys in 'embeddings' don't break.
            # But wait, SQLite might not let us insert explicit IDs easily if it's autoincrement without dropping everything.
            # Actually, `INSERT INTO identities (id, name, category)` works perfectly in SQLite.
            c.execute("INSERT INTO identities (id, name, category) VALUES (?, ?, ?)", (old_id, name, category))
            
        # Drop old table
        c.execute("DROP TABLE old_identities_v2")
        print("V3 Migration complete!")
        
    def _fix_foreign_keys(self, conn):
        c = conn.cursor()
        print("Fixing broken foreign keys in embeddings table...")
        c.execute('PRAGMA foreign_keys = OFF')
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
        print("Foreign keys fixed!")

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

    def merge_identities(self, source_id, target_id):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('UPDATE embeddings SET identity_id = ? WHERE identity_id = ?', (target_id, source_id))
        c.execute('DELETE FROM identities WHERE id = ?', (source_id,))
        conn.commit()
        conn.close()

    def log_detection(self, identity_id, name, category):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('INSERT INTO detection_logs (identity_id, name, category) VALUES (?, ?, ?)', (identity_id, name, category))
        conn.commit()
        conn.close()
        
    def get_detection_logs(self, category_filter=None, name_search=None):
        conn = self.get_connection()
        c = conn.cursor()
        
        query = 'SELECT id, name, category, timestamp FROM detection_logs WHERE 1=1'
        params = []
        
        if category_filter and category_filter != 'All':
            query += ' AND category = ?'
            params.append(category_filter)
            
        if name_search:
            query += ' AND name LIKE ?'
            params.append(f'%{name_search}%')
            
        query += ' ORDER BY timestamp DESC'
        c.execute(query, params)
        rows = c.fetchall()
        
        logs = []
        for row in rows:
            # Safely format timestamp string if it's already a string from sqlite
            ts = row[3]
            if hasattr(ts, 'strftime'):
                ts = ts.strftime('%Y-%m-%d %H:%M:%S')
            logs.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'timestamp': ts
            })
            
        conn.close()
        return logs

    def delete_detection_log(self, log_id):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('DELETE FROM detection_logs WHERE id = ?', (log_id,))
        conn.commit()
        conn.close()
        
    def clear_all_detection_logs(self):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('DELETE FROM detection_logs')
        conn.commit()
        conn.close()
