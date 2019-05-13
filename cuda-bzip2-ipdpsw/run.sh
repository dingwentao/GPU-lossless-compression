#!/bin/bash

domake=0
docheck=0

while getopts "mc" opt
do

case $opt in

m)

domake=1
;;

c)

docheck=1
;;

\?)
exit
;;

esac 
done

shift $(( OPTIND-1 ))

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]
then
echo "usage: ./run.sh [-m] [-c] <blocksize> <filename>"
echo "-m: make, -c: check"
exit
fi

blocksize=$1
threads=$2
filename=$3


if [ $domake -eq 1 ]
then
make clean
make bzip2
echo "*****************"
fi

echo "*****************"
echo "Compressing file "$filename
rm -f $filename".bz2"
./bzip2 -kf -$blocksize -n$threads $filename
ls -lh $filename".bz2"
ls -l $filename".bz2"
sizeVal=`ls -l $filename.bz2 | cut -d' ' -f5`
echo $sizeVal
echo $sizeVal"/(1024*1024)" | bc -l
echo "Done compressing"
echo "*****************"

if [ $docheck -eq 1 ]
then
mv $filename diff.txt
./bzip2 -d $filename".bz2"
echo -n "Performing diff .... "
diff diff.txt $filename
if [ $? -eq 0 ]
then
echo "Success"
else
echo "Failed"
mv diff.txt $filename
fi
fi
echo "***************"
