#!/bin/bash

VERSION="0.0.1"

# Use the old Dockerfile build process
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Get the absolute locations
SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=${SCRIPT_DIR}/..

# Image name is the folder name
IMAGE_NAME=$(basename $SCRIPT_DIR)

# Get the paramters
CACHE=" "
while getopts n flag
do
    case "${flag}" in
        n) CACHE=" --no-cache";;
    esac
done

clean () {
#    rm -R -f ${SCRIPT_DIR}/src
    rm -R ${SCRIPT_DIR}/environment.yml
}


# Identifying build version
cp ${ROOT_DIR}/environment.yml ${SCRIPT_DIR}

echo "Building image ${IMAGE_NAME}:${VERSION}..."
#docker build --cache-from ${REGISTRY}/${IMAGE_NAME}:${VERSION} -t ${IMAGE_NAME}:${VERSION} ${SCRIPT_DIR}

# Building image according to architecture
BUILD="build"
BUILD_ARGS=""
if [[ `uname -m` == 'arm64' ]]; then
  BUILD_ARGS="--build-arg ARCH=arm64v8/"
  echo "Building for ARM Processor"
fi

docker ${BUILD} ${CACHE} -t ${IMAGE_NAME}:${VERSION} ${SCRIPT_DIR} ${BUILD_ARGS}

if [ $? == 0 ]; then

    echo "Build successful: ${IMAGE_NAME}"
    clean
    exit 0

else
    echo "ERR: Build failed for ${IMAGE_NAME}"
    clean
    exit 1
fi

