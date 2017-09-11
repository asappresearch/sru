
Code used for speech recognition tasks. We evaluate LSTM and SRU on SWBD task.

## How to run
  - Build Kaldi from [Kaldi](https://github.com/kaldi-asr/kaldi.git).
  - Build CNTK from  [yzhang87/cntk](https://github.com/yzhang87/CNTK.git).
  - Build KaldiReader in CNTK (follow the instruction).
  - Build SWBD baseline (Set proper path in **run.sh**):
  ```
    run.sh
  ```
  - Run **run_SRU.sh** and get the results.
  ```
    run_SRU.sh --ndl ndlfile
  ```
 
  <br>
  
## TODO
   - Make it compatible with newest CNTK version.
   - Write custimized kernel for SRU
   - Port to Pytorch and MXNet 
