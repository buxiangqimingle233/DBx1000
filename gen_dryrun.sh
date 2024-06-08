#!/bin/bash

rm ./rundb*
rm -rf sniper-results/*
rm -rf host-results/*
rm ./*.png
rm ./*.log
rm ./*.out
make clean
docker exec docker_sniper-dev-container_1 pkill -9 -f sniper
