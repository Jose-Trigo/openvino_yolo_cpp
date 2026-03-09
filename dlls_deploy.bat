@echo off
echo Collecting required DLLs...

set BUILD_DIR=build
set DEPLOY_DIR=deploy
set OV_BIN=C:\Intel\openvino_2026\runtime\bin\intel64\Release
set TBB_BIN=C:\Intel\openvino_2026\runtime\3rdparty\tbb\bin
set CV_BIN=C:\libs\opencv\build\x64\vc16\bin

mkdir %DEPLOY_DIR% 2>nul

REM Copy executable
copy %BUILD_DIR%\ov_yolo_video.exe %DEPLOY_DIR%\

REM Copy OpenVINO DLLs
copy "%OV_BIN%\openvino.dll" %DEPLOY_DIR%\
copy "%OV_BIN%\openvino_c.dll" %DEPLOY_DIR%\
copy "%OV_BIN%\openvino_intel_cpu_plugin.dll" %DEPLOY_DIR%\

REM Copy TBB DLLs
copy "%TBB_BIN%\tbb12.dll" %DEPLOY_DIR%\
copy "%TBB_BIN%\tbbmalloc.dll" %DEPLOY_DIR%\

REM Copy OpenCV DLLs
copy "%CV_BIN%\opencv_world4130.dll" %DEPLOY_DIR%\
copy "%CV_BIN%\opencv_videoio_ffmpeg4130_64.dll" %DEPLOY_DIR%\

REM Copy model files
xcopy /E /I yolo_nano_v2_1_class_640_no_filter_int8_openvino_model %DEPLOY_DIR%\yolo_nano_v2_1_class_640_no_filter_int8_openvino_model

REM Copy video file (optional)
if exist brt_presentation.mp4 copy brt_presentation.mp4 %DEPLOY_DIR%\

echo Done! Deployment package in %DEPLOY_DIR%\
pause