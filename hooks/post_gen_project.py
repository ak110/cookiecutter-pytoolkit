#!/usr/bin/env python3
import subprocess


def _main():
    # pytoolkitを追加してInitial commit
    subprocess.run("git init", shell=True, check=True)
    subprocess.run(
        "git submodule add https://github.com/ak110/pytoolkit.git",
        shell=True,
        check=True,
    )
    subprocess.run("git add -A", shell=True, check=True)
    subprocess.run('git commit -m "Initial commit."', shell=True, check=True)
    # git commit時にflake8する設定
    subprocess.run("flake8 --install-hook=git", shell=True, check=True)
    subprocess.run("flake8 --install-hook=git", shell=True, check=True, cwd="pytoolkit")


if __name__ == "__main__":
    _main()
