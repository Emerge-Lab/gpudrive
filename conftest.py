import subprocess
import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def run_ctest():
    build_dir = "build"  # Replace this with the path to your build directory
    print("Running ctest in", os.path.abspath(build_dir))

    # Run ctest in the build directory
    try:
        result = subprocess.run(["ctest"], check=True, cwd=build_dir, capture_output=True, text=True)
        print("ctest stdout:")
        print(result.stdout)
        print("ctest stderr:")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ctest failed with error code {e.returncode}")
        print("Error output:")
        print(e.output)
        raise  # Re-raise the exception to make pytest fail
