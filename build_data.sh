#! /bin/bash 
rm -rf data
rm -rf data/real
rm -rf data/test_phish
rm -rf data/train_phish
rm -rf data/test_real
rm -rf data/train_real
rm -rf data/demo_data

mkdir data
mkdir data/real
mkdir data/test_phish
mkdir data/train_phish
mkdir data/test_real
mkdir data/train_real
mkdir data/demo_data

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

mv data/phishing3 data/demo_data # Move phishing data out to avoid including in test/train data
cp -r mbox/prof_emails data/demo_data # Data folder wiped at start, so move prof emails back to main data
rm data/real/real03951 # Remove remaining Enron data
rm data/real/real00000 # Remove blank first file

COUNT=0
for f in data/p*/* ; do
    if [ $COUNT -lt 3160 ]
    then
       mv $f data/train_phish/phish$COUNT
    else
       mv $f data/test_phish/phish$((COUNT-3160))
    fi
    COUNT=$((COUNT+1))
done

COUNT=0

for f in data/real/* ; do
   if [ $COUNT -lt 3160 ]
   then
      mv $f data/train_real/real$COUNT
   else
      mv $f data/test_real/real$((COUNT-3160))
   fi
   COUNT=$((COUNT+1))
done
echo Data build complete...