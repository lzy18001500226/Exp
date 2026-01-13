@echo off
echo Activating conda environment 'dronerfa'...
call conda activate dronerfa

echo Starting S3R training on DroneRFb-Spectra dataset...
python train.py

pause
