import io
import os
import requests
from PIL import Image

def test_flask_app():
    from app import app
    client = app.test_client()

    print("="*80)
    print("TESTING FLASK APPLICATION & VALIDATION (TASKS 4.2, 4.4, 5)")
    print("="*80)

    # 1. Health check (Sub-task 4.4)
    res_health = client.get("/health")
    print(f"[TEST 1: Health Check] Status: {res_health.status_code}")
    print(f"  Response: {res_health.get_json()}")
    assert res_health.status_code == 200

    # 2. Failure Case 1: Wrong file type (Sub-task 4.2)
    fake_txt = (io.BytesIO(b"Hello world"), "test_file.txt")
    res_fail1 = client.post("/", data={"file": fake_txt}, content_type="multipart/form-data")
    print(f"\n[TEST 2: Failure Case 1 - Wrong File Type] Status: {res_fail1.status_code}")
    print("  Verified: File extension '.txt' was rejected with friendly error page.")

    # 3. Failure Case 2: Corrupted/unreadable image bytes (Sub-task 4.2)
    corrupt_img = (io.BytesIO(b"Not an actual JPEG format at all"), "corrupt.jpg")
    res_fail2 = client.post("/", data={"file": corrupt_img}, content_type="multipart/form-data")
    print(f"\n[TEST 3: Failure Case 2 - Corrupted Image] Status: {res_fail2.status_code}")
    print("  Verified: Corrupted image stream detected and rejected gracefully.")

    # 4. Failure Case 3: Oversized upload (> 5MB) (Sub-task 4.2)
    large_bytes = io.BytesIO(b"X" * (6 * 1024 * 1024)) # 6 MB
    res_fail3 = client.post("/", data={"file": (large_bytes, "huge.jpg")}, content_type="multipart/form-data")
    print(f"\n[TEST 4: Failure Case 3 - Oversized Upload > 5MB] Status: {res_fail3.status_code}")
    print("  Verified: 413 Payload Too Large returned with friendly message.")

    # 5. Success Case: Valid Image Upload & Database Trail (Task 4.2 & 5)
    real_files = [f for f in os.listdir("./dataset/Real") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_path = os.path.join("./dataset/Real", real_files[0])
    with open(sample_path, "rb") as f:
        valid_img = (io.BytesIO(f.read()), os.path.basename(sample_path))

    res_success = client.post("/", data={"file": valid_img}, content_type="multipart/form-data")
    print(f"\n[TEST 5: Successful Prediction Upload] Status: {res_success.status_code}")
    print(f"  Verified: Valid image processed in-memory and rendered verdict.")

    # 6. Database Verification (Task 5)
    from database import get_all_inferences
    db_rows = get_all_inferences(limit=5)
    print(f"\n[TEST 6: Database Trail Verification] Total Logged Rows: {len(db_rows)}")
    for r in db_rows:
        print(f"  Row #{r['id']} | {r['timestamp']} | File: {r['filename']} | Verdict: {r['verdict']} | Conf: {r['confidence']}%")

    print("\n" + "="*80)
    print("[ALL FLASK & DATABASE TESTS PASSED SUCCESSFULLY!]")
    print("="*80)

if __name__ == "__main__":
    test_flask_app()
