# We will use Ubuntu for our image
FROM ${ARCH}ubuntu:20.04 as base
ARG ARCH

ENV LANG=C.UTF-8
ENV TZ=UTC
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade

# Adding tools
RUN apt-get install -y wget bzip2 git sudo
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install -y libsasl2-dev python-dev libldap2-dev libssl-dev


# Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/

#RUN wget -O Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-aarch64.sh

# Anaconda installing
RUN if [ "${ARCH}" = "arm64v8/" ]; then \
wget -O Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-aarch64.sh; \
else \
wget -O Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh; \
fi

RUN bash Miniconda.sh -p /home/ubuntu/miniconda -b
RUN rm Miniconda.sh

# Set path to conda
ENV PATH /home/ubuntu/miniconda/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
#RUN conda update anaconda
#RUN conda update --all

RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    autoconf automake libtool \
    libleptonica-dev \
    zlib1g-dev \
    python3-dev \
    libffi-dev \
    ca-certificates \
    curl


RUN sudo apt-get update && sudo apt-get install vim -y --fix-missing

ENV PYTHONDONTWRITEBYTECODE=true

# collect static files
COPY environment.yml .
RUN conda env update --file environment.yml


RUN conda clean -afy \
    && find /home/ubuntu/miniconda/ -follow -type f -name '*.a' -delete \
    && find /home/ubuntu/miniconda/ -follow -type f -name '*.pyc' -delete \
    && find /home/ubuntu/miniconda/ -follow -type f -name '*.js.map' -delete

RUN sudo apt-get purge -y --auto-remove libsasl2-dev python-dev libldap2-dev libssl-dev build-essential



CMD ["/bin/bash"]


