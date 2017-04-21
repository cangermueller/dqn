#!/usr/bin/env bash

set -e
shopt -s extglob

check=1
function run {
  cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  eval $cmd
  if [ $check -ne 0 -a $? -ne 0 ]; then
    1>&2 echo "Command failed!"
    exit 1
  fi
}

cmd="./main.py
  --env FrozenLake-v0

  --nb_episode 100
  --nb_pretrain_step 100
  --experience_size 100

  --learning_rate 0.01
  --target_rate 1.0
  --batch_size 16
  "
run $cmd
