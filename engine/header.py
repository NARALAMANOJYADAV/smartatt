class ENUM_MOCK:
    def __init__(self, val=1):
        self.value = val

class ENGINE_MODE:
    M_IDENTIFY = ENUM_MOCK(1)
    M_ENROLL = ENUM_MOCK(2)

class ENGINE_CODE:
    E_NO_FILE = ENUM_MOCK(-1)
    E_NO_FACE = ENUM_MOCK(-2)

class LIVENESS_CODE:
    L_SPOOF = ENUM_MOCK(0)
    L_REAL = ENUM_MOCK(1)
    L_TOO_SMALL_FACE = ENUM_MOCK(2)
    L_TOO_LARGE_FACE = ENUM_MOCK(3)
    L_NO_FACE = ENUM_MOCK(4)
    L_LIVENESS_CHECK_FAILED = ENUM_MOCK(5)

import cv2
import numpy as np

class MockAttribute:
    def __init__(self, image_mat):
        self.x1 = 10
        self.y1 = 10
        if image_mat is not None:
            self.x2 = image_mat.shape[1] - 10
            self.y2 = image_mat.shape[0] - 10
            # Create a 4x4 pseudo-embedding by resizing the image
            gray = cv2.cvtColor(image_mat, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (8, 8))
            self.feature = small.flatten().astype(np.float32) / 255.0
        else:
            self.x2 = 100
            self.y2 = 100
            self.feature = np.zeros((64,), dtype=np.float32)

        self.liveness = 1
        self.gender = 1
        self.age = 25
        self.glass = 0
        self.mask = 0

def init_sdk():
    return 0

def detect_face(image_mat, param, mode):
    if image_mat is None:
        return -1, None
    return 1, [MockAttribute(image_mat)]

def get_similarity(f1, f2):
    if f1 is None or f2 is None:
        return 0.0
    # Calculate cosine similarity
    dot = np.dot(f1, f2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = dot / (norm1 * norm2)
    # The pseudo-embeddings might be very similar, so we stretch the scale a bit
    return float(similarity)
