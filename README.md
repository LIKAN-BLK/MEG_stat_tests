# MEG_stat_tests
This branch make statistics test on MEG data  

This test needs long epochs [-4300,500]ms. In this branch performed three types of test 
1.fix_vs_nonfix
  This test performed on "ball chosen SI" and "error ball chosen" data independently. 
  Program caclulates time-frequency representation of data. Then program devides epoch on nonoverlaping windows 300 ms long
  (-4300:-4000,-4000:-3700,...,200:500)
  The last one window [200:500] is fixation window, all others - non-fixation windows.
  Then program caclucaltes mean by time for each channel and frequency. Then we calculate t-statistics between fixation window and one
  of the non-fixation windows. Then performed FDR correction for t-values. T-values visualised on head-like topographies. 
  "heads" for different non-fix windows for one frequency placed on one picture. Different pictures for different windows
2. target_vs_nontarget_no_baseline
  This kind of test performed needs  "ball chosen SI" and "error ball chosen" data together. 
  Then program caclucaltes mean by time for each channel and frequency. Then we calculate t-statistics between "ball chosen SI" 
  and "error ball chosen" .  Then performed FDR correction for t-values. T-values visualised on head-like topographies. 
  "heads" for different freqs visualised on one picture for one experiment.
3. target_vs_nontarget_no_baseline
  Version of previos experiment but with baseline subtraction from time-frequency representation of the data/  Baseline  - [200:500]
  
  
  
