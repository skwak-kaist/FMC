#!/bin/bash

dataset="validation_original"

find ${dataset} -name '*.*' -exec basename {} \; > ./file_list.txt

exec 10<&0

exec < file_list.txt 

exec 0<&10 10<&-


