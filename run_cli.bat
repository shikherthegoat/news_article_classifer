@echo off
echo ===================================================
echo  News Article Classifier — Interactive CLI
echo ===================================================
call "%~dp0venv\Scripts\activate.bat"
python "%~dp0demo_cli.py" %*
pause
