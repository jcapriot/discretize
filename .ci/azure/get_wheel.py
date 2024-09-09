import pathlib
from glob import glob
from packaging.tags import sys_tags, parse_tag
import sys


def get_wheel(wheel_dir: pathlib.Path):
    wheels = glob(str(wheel_dir / "*.whl"))

    # Get the current platform's tags
    current_platform_tags = list(sys_tags())

    for wheel in wheels:
        try:
            # Extract the tag portion of the wheel filename
            # For example: 'pkg-1.0.0-py3-none-any.whl'
            tags_part = wheel.split("-")[-3:]
            wheel_tags = parse_tag("-".join(tags_part))

            # Check if any of the wheel's tags match the current platform's tags
            if any(w_tag in current_platform_tags for w_tag in wheel_tags):
                return wheel
        except Exception as e:
            print(f"Error parsing {wheel}: {e}")
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise TypeError("Call signature is python get_wheel.py wheel_dir")
    wheel_dir = sys.argv[1]
    wheel_dir = pathlib.Path(wheel_dir)
    if not wheel_dir.is_dir():
        raise ValueError(f"{wheel_dir} is not a directory.")

    valid_wheels = get_wheel(wheel_dir)
    print(valid_wheels)
