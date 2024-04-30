FROM gitpod/workspace-full-vnc

RUN apt-get update && apt-get install -yq \
        libgl1-mesa-glx \
&& apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*
