FROM python:3.11-slim-bookworm

# System packages:
#   libgl1, libglib2.0-0, libsm6, libxext6  - needed by opencv-python to load
#                                            and to open windows
#   git, make                                - version control and the Makefile
#   rsync, openssh-client                    - copying files to/from the Pi
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 git make rsync openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# A regular user with the same user ID (1000) as the default WSL user, so files
# created inside the container aren't owned by root on your WSL side.
RUN useradd --create-home --uid 1000 dev
USER dev
WORKDIR /workspace