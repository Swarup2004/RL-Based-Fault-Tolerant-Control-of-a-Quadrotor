@echo off
REM ──────────────────────────────────────────────────────────────
REM Quadrotor FTC: Interactive 3D flight viewer
REM Keyboard controls:
REM   F  -- inject fault on current motor
REM   1-4 -- select which motor to fault
REM   R  -- reset episode
REM   Space -- pause/resume
REM   Q  -- quit
REM ──────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo.
echo ==========================================
echo  Interactive 3D flight viewer
echo ==========================================
python -u interactive_viz.py --model ./checkpoints/level_3_severe_fault

echo.
echo Done.
