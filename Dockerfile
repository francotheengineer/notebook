FROM tensorflow/tensorflow:latest

RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

RUN apt-get install software-properties-common && \
    add-apt-repository ppa:george-edison55/cmake-3.x && \
    apt-get update && \
    apt-get install -y libboost-dev libboost-python-dev cmake
RUN pip install dlib pillow matplotlib scikit-image
