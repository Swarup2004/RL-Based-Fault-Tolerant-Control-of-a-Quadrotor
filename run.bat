@echo off
REM ──────────────────────────────────────────────────────────────
REM Quadrotor FTC: train and evaluate
REM Usage:
REM   run.bat              -> evaluate pre-trained model
REM   run.bat train        -> train from scratch (full curriculum)
REM   run.bat resume       -> resume Level 3 from included checkpoint
REM ──────────────────────────────────────────────────────────────

cd /d "%~dp0"
set MODE=%1
if "%MODE%"=="" set MODE=eval

if "%MODE%"=="train" goto TRAIN
if "%MODE%"=="resume" goto RESUME
goto EVAL

:EVAL
echo.
echo ==========================================
echo  Evaluating Level 3 model (100 episodes)
echo ==========================================
python -u eval.py --model ./checkpoints/level_3_severe_fault --level 3 --episodes 100
goto END

:TRAIN
echo.
echo WARNING: Training from scratch will overwrite the included checkpoint.
echo Back up the checkpoints folder if you want to keep the pre-trained model.
echo.
echo ==========================================
echo  Training Level 0: healthy hover
echo ==========================================
python -u train.py --level 0 --n-envs 4
if errorlevel 1 goto ERROR

echo.
echo ==========================================
echo  Training Level 1: mild fault
echo ==========================================
python -u train.py --resume ./checkpoints/level_0_healthy --level 1 --n-envs 4
if errorlevel 1 goto ERROR

echo.
echo ==========================================
echo  Training Level 2: moderate fault
echo ==========================================
python -u train.py --resume ./checkpoints/level_1_mild_fault --level 2 --n-envs 4
if errorlevel 1 goto ERROR

echo.
echo ==========================================
echo  Training Level 3: severe fault
echo ==========================================
python -u train.py --resume ./checkpoints/level_2_moderate_fault --level 3 --n-envs 4
if errorlevel 1 goto ERROR

echo.
echo ==========================================
echo  Evaluating final model
echo ==========================================
python -u eval.py --model ./checkpoints/level_3_severe_fault --level 3 --episodes 100
goto END

:RESUME
echo.
echo ==========================================
echo  Resuming Level 3 from included checkpoint
echo ==========================================
python -u train.py --resume ./checkpoints/level_3_severe_fault --level 3 --n-envs 4
if errorlevel 1 goto ERROR

echo.
echo ==========================================
echo  Evaluating
echo ==========================================
python -u eval.py --model ./checkpoints/level_3_severe_fault --level 3 --episodes 100
goto END

:ERROR
echo.
echo ERROR: command failed. Check output above.
exit /b 1

:END
echo.
echo Done.
