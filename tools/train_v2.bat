@echo off
rem Full v2 training (12 epochs, 640px, CPU). Pass "resume" to continue an interrupted run.
cd /d "%~dp0.."
set MODE=%1
if "%MODE%"=="" set MODE=new
".venv-dml\Scripts\python.exe" src\train_v2.py 12 640 cpu %MODE% 6 >> "outputs\train_v2_stdout.log" 2>> "outputs\train_v2_stderr.log"