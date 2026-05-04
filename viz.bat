@echo off
REM ──────────────────────────────────────────────────────────────
REM Quadrotor FTC: visualization scripts
REM Usage:
REM   viz.bat              -> trajectory plots (saves to PNG)
REM   viz.bat compare      -> RL vs PID comparison plot (saves to PNG)
REM   viz.bat interactive  -> interactive 3D flight viewer
REM ──────────────────────────────────────────────────────────────

cd /d "%~dp0"
set MODE=%1
if "%MODE%"=="" set MODE=plot

if "%MODE%"=="compare" goto COMPARE
if "%MODE%"=="interactive" goto INTERACTIVE
goto PLOT

:PLOT
echo.
echo ==========================================
echo  Trajectory plots (single episode)
echo ==========================================
set MPLBACKEND=Agg
python -u visualize.py --model ./checkpoints/level_3_severe_fault --level 3 --save
echo Saved: trajectory_episode_1.png trajectory_episode_2.png
goto END

:COMPARE
echo.
echo ==========================================
echo  Trajectory comparison: RL vs PID
echo ==========================================
set MPLBACKEND=Agg
python -u visualize.py --model ./checkpoints/level_3_severe_fault --level 3 --compare --save
echo Saved: trajectory_compare.png
goto END

:INTERACTIVE
echo.
echo ==========================================
echo  Interactive 3D flight viewer
echo ==========================================
python -u interactive_viz.py --model ./checkpoints/level_3_severe_fault
goto END

:END
echo.
echo Done.
