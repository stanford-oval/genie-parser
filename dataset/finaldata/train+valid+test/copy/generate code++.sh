#!/bin/bash

cut -f3 ./train.txt > ./code++.txt
cut -f3 ./valid.txt >> ./code++.txt
cut -f3 ./test.txt >> ./code++.txt