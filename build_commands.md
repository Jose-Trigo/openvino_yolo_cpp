open this cmd for cl visual studio build tools compiler

type in windows serach bar " x64 native tools command prompt for vs "


cd C:\Users\jpedr\Repos_2nd_year\openvino_yolo_cpp
rmdir /S /Q build
mkdir build
cd build

call "C:\Program Files (x86)\Intel\openvino_2024\setupvars.bat"

set OpenCV_DIR=C:\libs\opencv\build
set PATH=C:\libs\opencv\build\x64\vc16\bin;%PATH%

cmake -D OpenCV_ARCH=x64 -D OpenCV_RUNTIME=vc16 -D OpenCV_STATIC=OFF -G "NMake Makefiles" ..
cmake --build .








###### PC EFACEC

cd C:\Users\11032\repos\openvino_yolo_cpp
rmdir /S /Q build
mkdir build
cd build

call "C:\Intel\openvino_2026\setupvars.bat"

set OpenCV_DIR=C:\libs\opencv\build 
set PATH=C:\libs\opencv\build\x64\vc16\bin;%PATH%

set PATH=C:\Users\11032\repos\openvino_yolo_cpp\third_party_dependencies\openh264;%PATH%

cmake -DCMAKE_BUILD_TYPE=Debug -D OpenCV_ARCH=x64 -D OpenCV_RUNTIME=vc16 -D OpenCV_STATIC=OFF -G "NMake Makefiles" .. 
cmake -DCMAKE_BUILD_TYPE=Release -D OpenCV_ARCH=x64 -D OpenCV_RUNTIME=vc16 -D OpenCV_STATIC=OFF -G "NMake Makefiles" ..
cmake -D OpenCV_ARCH=x64 -D OpenCV_RUNTIME=vc16 -D OpenCV_STATIC=OFF -G "NMake Makefiles" ..
cmake --build .

.\build\ov_yolo_video.exe










#### Runing this on another pc no msvc compiler needed

## for now its the poor mans solution, you need the opencv libs, the openvinno runtime libs, and the openh264 dll

just run these commands, from a cmd

cd C:\Users\11032\repos\openvino_yolo_cpp

call C:\Intel\openvino_2026\setupvars.bat

set OpenCV_DIR=C:\libs\opencv\build
set PATH=C:\libs\opencv\build\x64\vc16\bin;%PATH%
set PATH=C:\Users\11032\repos\openvino_yolo_cpp\third_party_dependencies\openh264;%PATH%

build\ov_yolo_video.exe

