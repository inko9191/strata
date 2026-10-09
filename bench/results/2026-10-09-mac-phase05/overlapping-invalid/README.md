# Invalid preliminary measurements

Two reference runners overlapped because the first runner was not stopped by the restart command.
The affected CLI logs include Metal command-buffer out-of-memory errors despite exit code zero.
These model timings and captures are not used in the final comparison or successful-run medians.
The diagnostic runner now holds an exclusive file lock. Strata and llama.cpp sources were not changed.

The preliminary bandwidth probe ran while model downloads were active: its attempted downloader pause
did not match the process. It was repeated after downloads finished and before the clean model run.
The final conditions record does not claim that the earlier downloader pause succeeded.
