#!/usr/bin/env python3
"""プロジェクト作成後の処理。"""
import os
import subprocess


def _main():
    url = "git@github.com:ak110/pytoolkit.git"
    url = os.environ.get("_PYTOOLKIT_URL", url)

    # pytoolkitを追加してInitial commit
    subprocess.run("git init", shell=True, check=True)
    subprocess.run(f"git submodule add {url} pytoolkit.git", shell=True, check=True)
    subprocess.run("ln -s pytoolkit.git/pytoolkit", shell=True, check=True)
    subprocess.run("git add -A", shell=True, check=True)
    subprocess.run('git commit -m "Initial commit."', shell=True, check=True)

    # pre-commit
    subprocess.run("pre-commit install", shell=True, check=False)
    subprocess.run("pre-commit install", shell=True, check=False, cwd="pytoolkit")


if __name__ == "__main__":
    _main()
