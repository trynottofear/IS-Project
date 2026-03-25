import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import cv2

class FaceProcessor:
    def __init__(self, db_manager):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        print(f"FaceProcessor running on device: {self.device}")
        
        # Initialize MTCNN for face detection (fast, GPU enabled if available)
        self.mtcnn = MTCNN(
            keep_all=True, device=self.device, min_face_size=40, thresholds=[0.6, 0.7, 0.7]
        )
        
        # Single face extractor for enrollment
        self.mtcnn_single = MTCNN(
            keep_all=False, device=self.device, min_face_size=40
        )
        
        # Initialize InceptionResnetV1 for face embedding (loads pretrained weights at first run)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.db = db_manager
        
        # Threshold for cosine similarity (can be tuned for stricter/looser matching)
        self.similarity_threshold = 0.75

    def get_embedding(self, img_pil):
        """ Extract embedding from a single face image. Used when enrolling a user. """
        face = self.mtcnn_single(img_pil)
        if face is not None:
            # Face returns as 3D tensor: (C, H, W)
            face = face.unsqueeze(0).to(self.device) # Add batch dimension
            with torch.no_grad():
                embedding = self.resnet(face)
            return embedding[0].cpu().numpy()
        return None

    def extract_faces_and_embeddings(self, frame_cv):
        """
        Detects faces and computes embeddings without DB matching.
        Returns a tuple (boxes, list of embeddings).
        """
        rgb_frame = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        boxes, probs = self.mtcnn.detect(img_pil)
        
        embeddings = []
        if boxes is not None:
            faces = self.mtcnn.extract(img_pil, boxes, save_path=None)
            if faces is not None:
                faces = faces.to(self.device)
                with torch.no_grad():
                    embeddings = self.resnet(faces).cpu().numpy()
        return boxes if boxes is not None else [], embeddings

    def process_frame(self, frame_cv, threshold=None):
        """
        Process a cv2 frame. Detect faces, compute embeddings, and match with DB.
        Returns a list of dicts with box, name, category, identity_id, similarity, embedding.
        """
        if threshold is None:
            threshold = self.similarity_threshold
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)

        # Detect faces
        boxes, probs = self.mtcnn.detect(img_pil)
        
        results = []
        if boxes is not None:
            # Extract faces as tensors
            faces = self.mtcnn.extract(img_pil, boxes, save_path=None)
            
            if faces is not None:
                faces = faces.to(self.device)
                with torch.no_grad():
                    embeddings = self.resnet(faces).cpu().numpy()
                
                # Fetch all identities to compare against
                identities = self.db.get_all_identities_with_embeddings()
                
                for box, embedding in zip(boxes, embeddings):
                    best_match_name = "Unknown"
                    best_match_category = "Unknown"
                    best_match_id = None
                    best_sim = -1.0
                    
                    # Normalize embedding for cosine similarity
                    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
                    
                    for identity in identities:
                        identity_best_sim = -1.0
                        
                        # Compare against all embeddings for this identity
                        for emb_dict in identity['embeddings']:
                            db_emb = emb_dict['embedding']
                            db_emb_norm = db_emb / (np.linalg.norm(db_emb) + 1e-8)
                            
                            sim = np.dot(emb_norm, db_emb_norm)
                            if sim > identity_best_sim:
                                identity_best_sim = sim
                                
                        if identity_best_sim > best_sim:
                            best_sim = identity_best_sim
                            if identity_best_sim > threshold:
                                best_match_name = identity['name']
                                best_match_category = identity['category']
                                best_match_id = identity['id']
                                
                    results.append({
                        'box': [int(b) for b in box],
                        'name': best_match_name,
                        'category': best_match_category,
                        'identity_id': best_match_id,
                        'similarity': float(best_sim),
                        'embedding': embedding
                    })
        return results
