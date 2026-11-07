import os
import sys
import subprocess
import shutil

DVC_EXE = os.path.join(sys.prefix, "Scripts", "dvc.exe")
if not os.path.exists(DVC_EXE):
    DVC_EXE = f'"{sys.executable}" -m dvc'

def run_cmd(cmd, cwd=None):
    # Replace bare 'dvc' with explicit python -m dvc or DVC_EXE
    if cmd.startswith("dvc "):
        cmd = f'"{sys.executable}" -m dvc ' + cmd[4:]
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if res.stdout.strip():
        print(f"[STDOUT]\n{res.stdout.strip()}")
    if res.stderr.strip() and res.returncode != 0:
        print(f"[STDERR]\n{res.stderr.strip()}")
    return res

def run_dvc_toy_pipeline(base_dir):
    toy_dir = os.path.join(base_dir, "dvc_toy_part")
    remote_storage_dir = os.path.join(base_dir, "dvc_remote_storage")

    if os.path.exists(toy_dir):
        shutil.rmtree(toy_dir, ignore_errors=True)
    if os.path.exists(remote_storage_dir):
        shutil.rmtree(remote_storage_dir, ignore_errors=True)

    os.makedirs(toy_dir, exist_ok=True)
    os.makedirs(remote_storage_dir, exist_ok=True)

    print("="*80)
    print("SUB-TASK 1.4: DVC TOY PART EXECUTION")
    print("="*80)

    # 1. Init Git & DVC
    run_cmd("git init", cwd=toy_dir)
    run_cmd('git config user.name "VisionGuard Student"', cwd=toy_dir)
    run_cmd('git config user.email "student@visionguard.org"', cwd=toy_dir)
    run_cmd("dvc init", cwd=toy_dir)
    run_cmd('git commit -m "Initialize Git and DVC"', cwd=toy_dir)

    # 2. Create small CSV v1
    csv_file = os.path.join(toy_dir, "dataset.csv")
    with open(csv_file, "w") as f:
        f.write("id,sample_name,label\n1,sample_001,Real\n2,sample_002,AI\n3,sample_003,Real\n")
    print("\n[+] Created dataset.csv (Version 1 - 3 rows)")

    # 3. Track with DVC
    run_cmd("dvc add dataset.csv", cwd=toy_dir)
    run_cmd("git add dataset.csv.dvc .gitignore", cwd=toy_dir)
    run_cmd('git commit -m "Track dataset.csv v1 with DVC"', cwd=toy_dir)

    # 4. Modify CSV (Version 2)
    with open(csv_file, "a") as f:
        f.write("4,sample_004,AI\n5,sample_005,Real\n6,sample_006,AI\n")
    print("\n[+] Modified dataset.csv (Version 2 - 6 rows)")

    # 5. Track v2 with DVC
    run_cmd("dvc add dataset.csv", cwd=toy_dir)
    run_cmd("git add dataset.csv.dvc", cwd=toy_dir)
    run_cmd('git commit -m "Update dataset.csv v2 with DVC"', cwd=toy_dir)

    # 6. Show Git log of commits
    print("\n" + "="*80)
    print("GIT COMMIT HISTORY SHOWING BOTH DVC DATA VERSIONS:")
    print("="*80)
    run_cmd("git log --oneline -n 3", cwd=toy_dir)

    # 7. Rollback to older version (v1)
    print("\n" + "="*80)
    print("ROLLING BACK TO OLDER VERSION (V1):")
    print("="*80)
    run_cmd("git checkout HEAD~1 dataset.csv.dvc", cwd=toy_dir)
    run_cmd("dvc checkout", cwd=toy_dir)
    with open(csv_file, "r") as f:
        print("[dataset.csv content after rollback to v1]:\n" + f.read().strip())

    # 8. Return to newer version (v2)
    print("\n" + "="*80)
    print("RETURNING TO NEWER VERSION (V2):")
    print("="*80)
    run_cmd("git checkout HEAD dataset.csv.dvc", cwd=toy_dir)
    run_cmd("dvc checkout", cwd=toy_dir)
    with open(csv_file, "r") as f:
        print("[dataset.csv content after returning to v2]:\n" + f.read().strip())

    # 9. Configure remote storage, push, delete local data, and pull it back
    print("\n" + "="*80)
    print("CONFIGURING REMOTE STORAGE, PUSHING DATA, AND PULLING:")
    print("="*80)
    run_cmd(f'dvc remote add -d local_remote "{remote_storage_dir}"', cwd=toy_dir)
    run_cmd("git add .dvc/config", cwd=toy_dir)
    run_cmd('git commit -m "Configure local DVC remote storage"', cwd=toy_dir)
    run_cmd("dvc push", cwd=toy_dir)

    # Delete local data
    os.remove(csv_file)
    print(f"\n[!] Deleted local file 'dataset.csv'. Exists locally? {os.path.exists(csv_file)}")

    # Pull back from remote
    run_cmd("dvc pull", cwd=toy_dir)
    print(f"[+] Restored local file 'dataset.csv' from DVC remote! Exists locally? {os.path.exists(csv_file)}")
    with open(csv_file, "r") as f:
        print("[dataset.csv restored content]:\n" + f.read().strip())

    # 10. Inspect the small pointer file (.dvc)
    dvc_pointer_file = os.path.join(toy_dir, "dataset.csv.dvc")
    print("\n" + "="*80)
    print("POINTER FILE CONTENT (dataset.csv.dvc):")
    print("="*80)
    with open(dvc_pointer_file, "r") as f:
        print(f.read().strip())
    print("="*80)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_dvc_toy_pipeline(current_dir)
