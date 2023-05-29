@echo off
cd .venv\Scripts
call activate.bat
cd ..\..
echo Running in the rec_env virtual environment.
python --version
python -c "import sys; print('Environment:', sys.exec_prefix)"
python flask_app.py