#! /bin/bash
args=("$@")
./email_cleaner_3000_xtra_fast.awk ${args[0]} > temp.txt
first=`sed -n '1p' temp.txt`
second=`sed -n '2p' temp.txt`
sed -n "${first},${second}p" ${args[0]} 
