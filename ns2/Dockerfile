# Pinned legacy Ubuntu environment for NS2 (v2.35)
FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    automake \
    libxt-dev \
    libx11-dev \
    libxmu-dev \
    gcc-4.8 \
    g++-4.8 \
    wget \
    git \
    python3 \
    python3-pip \
    gawk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Expect ns-allinone-2.35.tar.gz in build context or download
RUN if [ ! -f ns-allinone-2.35.tar.gz ]; then \
        wget -q https://sourceforge.net/projects/nsnam/files/allinone/ns-allinone-2.35/ns-allinone-2.35.tar.gz; \
    fi && \
    tar xzf ns-allinone-2.35.tar.gz

ENV CC=gcc-4.8 CXX=g++-4.8

# Known fix for gcc >= 4.7 in linkstate/ls.h:
RUN sed -i 's/erase(baseMap::begin(), baseMap::end())/this->erase(baseMap::begin(), baseMap::end())/' /opt/ns-allinone-2.35/ns-2.35/linkstate/ls.h

# Copy patches
COPY patches/bodyshadow.h /opt/ns-allinone-2.35/ns-2.35/mobile/
COPY patches/bodyshadow.cc /opt/ns-allinone-2.35/ns-2.35/mobile/
COPY patches/wireless-phy-rssi.patch /opt/ns-allinone-2.35/ns-2.35/

# Apply patches and compile
WORKDIR /opt/ns-allinone-2.35/ns-2.35
RUN patch -p1 < wireless-phy-rssi.patch && \
    sed -i '/OBJ_CC =/ s/$/ mobile\/bodyshadow.o/' Makefile.in && \
    cd /opt/ns-allinone-2.35 && \
    ./install

ENV PATH="/opt/ns-allinone-2.35/bin:/opt/ns-allinone-2.35/tcl8.5.10/unix:/opt/ns-allinone-2.35/tk8.5.10/unix:${PATH}"
ENV LD_LIBRARY_PATH="/opt/ns-allinone-2.35/otcl-1.14:/opt/ns-allinone-2.35/lib:${LD_LIBRARY_PATH}"
ENV TCL_LIBRARY="/opt/ns-allinone-2.35/tcl8.5.10/library"

WORKDIR /work
CMD ["/bin/bash"]
