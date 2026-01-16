TODO
- "He" init instead of Xavier
- Increase model capacity
- Improve LR scheduling

TODO perf
- prioritizing functional/structural hotspots when masking
- embedding dim







1/11/26
- 500 epochs, experimented with the LR (4e-4 seems solid), exhausted model capacity... at 4 layers
- 0.1877 accuracy **revisited: I'm not confident that model capacity was necessarily exhausted


1/14/26 
 - 597 epochs, switched file type mid way through, 8 layers - not yet exhausted... 
 - train loss: 2.77, val loss: 2.73, accuracy: 0.1894


1/15/26
- Switched to sin pos encoder -> significant improvemements
- 17 epochs in -> 0.2 accuracy 2.73 train loss, 2.7 val loss -> (re)pause model training in-hopes of LR improvement (decay)
- 24 epochs in -> 0.204 -> increase LR to 8
- 26 epochs end -> 0.205
