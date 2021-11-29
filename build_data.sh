#! /bin/bash 
rm -rf data
rm -rf data/real
mkdir data
mkdir data/real
for f in mbox/*.mbox ; do
    name=${f#*/}
    name=${name%.*}
    mkdir data/$name/
    csplit -k -fdata/$name/phish -n4 $f '/^From /' {99999}
done

for f in data/*/* ; do
    echo 1 > $f.clean
    ./scripts/clean_file.awk $f >> $f.clean
    mv $f.clean $f
done

csplit -s -k -fdata/real/real -n5 mbox/emails.csv '/^Date: /' {9999}

for f in data/real/* ; do
   echo 0 > $f.clean
   sed '$d;1d' $f >> $f.clean
   mv $f.clean $f
done
