"""模块用途：保存 JSON、文件指纹与运行环境；不读写观测数据。"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'pandas', 'torch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = {'commit': None, 'dirty': None}
    repo = Path(__file__).resolve().parents[1]
    try:
        git['commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip()
        git['dirty'] = bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git['error'] = 'Git metadata unavailable'
    return {'versions': versions, 'git': git, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
