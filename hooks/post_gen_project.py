#!/usr/bin/env python3
import subprocess


def _main():
    subprocess.run("git init", shell=True, check=True)
    subprocess.run(
        "git submodule add https://github.com/ak110/pytoolkit.git",
        shell=True,
        check=True,
    )
    subprocess.run("git add -A", shell=True, check=True)
    subprocess.run(
        "git commit -m \"Initial commit.\"",
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    _main()
