import sys

def check_package(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

if __name__ == "__main__":
    required_packages = ["geopandas", "shapely", "pyproj"]
    missing_packages = []

    print("Checking for required packages:")
    for package in required_packages:
        if check_package(package):
            print(f"- {package}: Installed")
        else:
            print(f"- {package}: Not Installed")
            missing_packages.append(package)

    if missing_packages:
        print("\nError: The following packages are not installed:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPlease install them using 'pip install <package_name>' in your virtual environment.")
        sys.exit(1)
    else:
        print("\nAll required packages are installed. Environment is ready.")
        sys.exit(0)
