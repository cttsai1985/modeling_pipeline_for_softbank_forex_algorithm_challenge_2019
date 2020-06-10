ENV=cttsai1985/ml_env_rapids:stable
# ENV=cttsai1985/ml_env_rapids:nightly
#ENV=ml_env_rapids

GPU_DEVICE='"device=0,1"'

SHM_SIZE=16G

RootSrcPath=${PWD}
DockerRootSrcPath=/root/src/

DataPath=${PWD}/input
DockerDataPath=/root/src/input

RootPort1=8800
DockerRootPort1=8888

RootPort2=6600
DockerRootPort2=6666

docker rm $(docker ps -a -q)

CMD="jupyter notebook --port ${DockerRootPort1} --ip=0.0.0.0 --allow-root --no-browser"
CMD=bash

docker run -i -t --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v $RootSrcPath:$DockerRootSrcPath -v $(readlink -f $DataPath):$DockerDataPath --shm-size $SHM_SIZE $ENV $CMD


