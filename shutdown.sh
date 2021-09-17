#/bin/bash

while :
do
    thread_num=`ps -e |grep $1 | wc -l`
    if [ $thread_num -eq 0 ]; then
       date >> shutdowntime.log
       shutdown
       exit
   else
       echo "Sleeping $2 second..."
       sleep $2
   fi
done