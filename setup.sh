# sudo apt-get update
# sudo apt-get install -y python3-opengl
# sudo apt-get install -y libgl1-mesa-glx
# pip install PyOpenGL PyOpenGL_accelerate


# Cài đặt các dependencies cần thiết cho Qt và X11
sudo apt-get update
sudo apt-get install -y \
    libxcb1 \
    libxcb1-dev \
    libx11-xcb1 \
    libx11-xcb-dev \
    libxcb-keysyms1 \
    libxcb-keysyms1-dev \
    libxcb-image0 \
    libxcb-image0-dev \
    libxcb-shm0 \
    libxcb-shm0-dev \
    libxcb-icccm4 \
    libxcb-icccm4-dev \
    libxcb-sync1 \
    libxcb-sync-dev \
    libxcb-xfixes0-dev \
    libxcb-shape0-dev \
    libxcb-randr0-dev \
    libxcb-render-util0 \
    libxcb-render-util0-dev \
    libxcb-glx0-dev \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev

# Cài đặt thêm Qt platform plugins
sudo apt-get install -y \
    qt5-default \
    qtbase5-dev \
    qttools5-dev-tools

# Nếu bạn đang sử dụng conda environment, cài đặt lại PyQt5
conda install -c anaconda pyqt