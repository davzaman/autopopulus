# !/bin/bash
guild stop -Fo queue
guild runs rm -Fo queue
# for pid in $(ps -ef | awk '/queue_main/ {print $2}'); do pkill $pid; done
for pid in $(ps -ef | awk '/queue_main/ {print $2}'); do kill -9 $pid; done
ps -ef | grep "ray::IDLE" | grep -v grep | awk '{print $2}' | xargs -r kill -9

#  for pid in $(ps -ef | awk '/op_main/ {print $2}'); do kill -9 $pid; done
#  for pid in $(ps -ef | awk '/autopopulus\/main.py/ {print $2}'); do kill -9 $pid; done