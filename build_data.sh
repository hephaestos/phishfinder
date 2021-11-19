#! /usr/bin/bash 
rm -rf data
mkdir data
for f in mbox/*.mbox ; do
    name=${f#*/}
    name=${name%.*}
    mkdir data/$name/
    csplit -k -fdata/$name/phish -n4 $f '/^From /' {99999}
done

for f in data/*/* ; do
    ./scripts/clean_file.awk $f > $f.clean
    mv $f.clean $f
done
