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
    ./scripts/clean_file.awk $f >> $f.clean
    mv $f.clean $f
done

csplit -s -k -fdata/real/real -n5 mbox/emails.csv '/^Date: /' {3950}

for f in data/real/* ; do
   sed '$d;1d' $f >> $f.clean
   mv $f.clean $f
done

rm data/real/real03951 # Remove remaining Enron data
rm data/real/real03951 # Remove blank first file
