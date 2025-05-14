import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": [
        "flask",
        "pandas",
        "openpyxl",
        "numpy",
        "openai",
        "werkzeug",
        "jinja2"
    ],
    "excludes": [],
    "include_files": [
        ("templates", "templates"),
        ("static", "static"),
        ("uploads", "uploads"),
    ]
}

# GUI applications require a different base on Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="Excel AI Assistant",
    version="1.0",
    description="Excel AI Assistant Application",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "app.py",
            base=base,
            target_name="ExcelAI.exe",
            icon="static/favicon.ico",  # If you have an icon file
        )
    ],
) 