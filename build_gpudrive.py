import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO)


def main():
    # Cloning the repository, although typically you would not do this in the build step
    # as the code should already be present. Including it just for completeness.
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive", "--force"]
    )

    # Create and enter the build directory
    if not os.path.exists("build"):
        os.mkdir("build")
    os.chdir("build")

    # Run CMake and Make
    subprocess.check_call(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"])
    subprocess.check_call(
        ["make", f"-j{os.cpu_count()}"]
    )  # Utilize all available cores

    # Going back to the root directory
    os.chdir("..")


if __name__ == "__main__":
    logging.info("Building the C++ code and installing the Python package")
    main()
