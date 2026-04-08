#!/bin/bash

Data_path="./validation_original"
copy_path="./validation"
list_file="./annotations_5k/detection_validation_input_5k.lst"

exec 10<&0
exec < ${list_file}

while read LINE; do
	echo $LINE
	name=$(basename -s .png $LINE)
	cp ${Data_path}/$name.jpg ${copy_path}/$name.jpg
done
exec 0<&10 10<&-

