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
