import pytest
import os
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_read_main():
    filename = 'data/test_images/0af0c1a38.jpg'
    # files={"file": ("filename", open(filename, "rb"), "image/jpeg")}
    # files={"file": open(filename, "rb")}
    
    with open(filename, mode='rb') as test_file:
        files = {"file": (os.path.basename(filename), test_file, "image/jpeg")}
        response = client.post("/predict", files=files)
    
    
    # data = MultipartEncoder(fields={'file': ('filename', open(filename, 'rb'), 'image/jpeg')})
    
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/jpeg'
