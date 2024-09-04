# PyTorch with CUDAイメージをベースにする
# FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# # 必要なパッケージをインストール
# RUN DEBIAN_FRONTEND=noninteractive apt-get update \
#     && apt-get install -y --no-install-recommends \
#        python3-pip \
#        ffmpeg \
#        git \
#        libsm6 \
#        libxext6 \
#        libxrender-dev \
#     && rm -rf /var/lib/apt/lists/*

# # pipをアップグレード
# RUN pip install --upgrade pip

# # 必要なPythonパッケージのインストール
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt

# # face-alignmentライブラリのインストール
# RUN pip install git+https://github.com/1adrianb/face-alignment

# # アプリケーションコードをコピー
# COPY . /app/
# WORKDIR /app

# # FastAPIサーバーを起動
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]



FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# 必要なパッケージをインストール
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# アプリケーションをコンテナにコピー
COPY . /app/
WORKDIR /app

# pipをアップグレード
RUN pip3 install --upgrade pip

# face-alignmentライブラリのインストール
RUN pip3 install git+https://github.com/1adrianb/face-alignment

# Cythonのビルドに必要なパッケージをインストール
RUN pip3 install numpy cython pythran

# Cythonビルドの最適化
RUN pip3 install --upgrade setuptools wheel

# 残りのパッケージのインストール
RUN pip3 install -r requirements.txt

# OpenCVの設定（必要に応じて追加）
ENV DISPLAY=:0

# FastAPIサーバーを起動
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]