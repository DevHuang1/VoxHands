# VoxHands reference deployment image.
#
# The Python runtime image cannot install system libraries, and MuJoCo's
# offscreen renderer needs a GL stack on a display-less host. This image adds
# the EGL/OSMesa libraries so `/api/camera` and `/api/vision` work in
# production, then installs the same pinned runtime as the native build.

FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    MUJOCO_GL=egl

# libegl1 + Mesa give EGL software rendering (llvmpipe); libosmesa6 is the
# fallback backend. libgl1/libglib2.0-0 satisfy MuJoCo's loader dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libegl-mesa0 \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-render.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-render.txt

COPY run.py ./
COPY voxhands ./voxhands
COPY frontend ./frontend

EXPOSE 8000

CMD ["python", "run.py"]
