#!/usr/bin/env bash

WORKDIR=${WORKDIR:="./logs"}

date=$(date +%y%m%d-%H%M%S)
postfix=$(openssl rand -hex 4)
logname="${date}-${postfix}"
export CONFIG_DIR=${WORKDIR}/configs-${logname}
export LOGDIR=${WORKDIR}/logdir-${logname}

catalyst-dl run -C catalyst_fpn.yml --logdir ${LOGDIR}  #--autoresume best