# !/bin/bash
guild stop -Fo queue
guild runs rm -Fo queue
# for pid in $(ps -ef | awk '/queue_main/ {print $2}'); do pkill $pid; done
for pid in $(ps -ef | awk '/queue_main/ {print $2}'); do kill -9 $pid; done
ps -ef | grep "ray::IDLE" | grep -v grep | awk '{print $2}' | xargs -r kill -9

#  for pid in $(ps -ef | awk '/op_main/ {print $2}'); do kill -9 $pid; done
#  for pid in $(ps -ef | awk '/autopopulus\/main.py/ {print $2}'); do kill -9 $pid; done

# Cleanup mlflow deleted runs
# they don't get delted if you delete them in the ui they just get a "deleted_time" attr added to the meta.yaml
for exp_id in $(ls mlruns); do for run_id in $(ls mlruns/$exp_id);do if $(cat mlruns/$exp_id/$run_id/meta.yaml | grep -q "deleted_time"); then rm -r mlruns/$exp_id/$run_id ; fi; done ; done

# for pid in $(ps -ef | awk '/autopopulus\/impute.py/ {print $2}'); do kill -9 $pid; done