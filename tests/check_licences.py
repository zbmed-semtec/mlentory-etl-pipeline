"""
This module checks the licenses of installed Python packages and prints a table of the packages and their licenses.

The `get_pkg_license` function retrieves the license information for a given package by parsing the package's metadata. The `print_packages_and_licenses` function prints a table of all installed packages and their licenses. The `if __name__ == "__main__":` block calls `print_packages_and_licenses` and prints the number of unique licenses found.
"""

import pkg_resources
import prettytable

unique_licenses = set()


def get_pkg_license(pkg):
    """
    Retrieve the license information for a given package.

    This function attempts to extract the license information from the package's metadata.
    It first tries to read from the 'METADATA' file, and if that fails, it falls back to
    reading from the 'PKG-INFO' file. The function also adds any found licenses to the
    global 'unique_licenses' set.

    Args:
        pkg (pkg_resources.Distribution): The package to retrieve the license for.

    Returns:
        str: The license information for the package, or "(Licence not found)" if no
             license information could be extracted.
    """

    try:
        lines = pkg.get_metadata_lines("METADATA")
    except:
        lines = pkg.get_metadata_lines("PKG-INFO")

    for line in lines:
        if line.startswith("License:"):
            values = line[9:].split("AND")
            for value in values:
                unique_licenses.add(value)

            return line[9:]
    return "(Licence not found)"


def print_packages_and_licenses():
    t = prettytable.PrettyTable(["Package", "License"])
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        license = get_pkg_license(pkg)
        t.add_row((str(pkg), license))
    print(t)


if __name__ == "__main__":
    print_packages_and_licenses()
    print(f"Unique licenses: {len(unique_licenses)}")
    result = " and ".join(unique_licenses)
    print(result)
