import pkg_resources
import prettytable

unique_licenses = set()

def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            values = line[9:].split('AND')
            for value in values:
                unique_licenses.add(value)
                
            return line[9:]
    return '(Licence not found)'

def print_packages_and_licenses():
    t = prettytable.PrettyTable(['Package', 'License'])
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        t.add_row((str(pkg), get_pkg_license(pkg)))
    print(t)


if __name__ == "__main__":
    print_packages_and_licenses()
    print(f'Unique licenses: {len(unique_licenses)}')
    result = " and ".join(unique_licenses)
    print(result)