"""Install the exact upstream PyRadiomics v3.1.0 source in the active environment.

Requires a C compiler (build-essential on Linux, Visual C++ Build Tools on Windows).
This works around the upstream release's inconsistent package-version metadata;
no scientific source code is patched. Run in an isolated environment.
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "numpy==1.26.4",
            "setuptools<72",
            "wheel",
            "versioneer==0.29",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-r",
            str(root / "requirements-radiomics.txt"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
