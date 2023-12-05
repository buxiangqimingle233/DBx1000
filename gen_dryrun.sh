#!/bin/bash

rm ./rundb*
rm -rf sniper-results/*
rm -rf host-results/*
rm ./*.png
make clean
docker exec docker_sniper-dev-container_1 pkill -9 -f sniper