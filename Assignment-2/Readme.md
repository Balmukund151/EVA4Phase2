*********************************************************************************************

Max Accuracy= 89.26% in 11th Epoch



*********************************************************************************************


1. Code Overview-

	a) dataloaders.py -  contains the dataloaders and transforms for train & test.
	
	b) graphs_and_other_utility_functions.py - contains the basic utility functions like plotting, getting LR etc required for execution
	
	c) train_test.py - contains training and test code for NN.
2. Resizing Strategy:- used albumentations.Resize (with cv2.INTER_LINEAR) as this worked best
3. model trained on :- mobilenet_v2
4. accuracy graph :-
	![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-2/Test_Accuracy_and_Test_Loss.png)
5. Misclassified Images from each category:-  

'Flying Birds'=0  'Large QuadCopters'=1 'Small QuadCopters'=2 'Winged Drones'=3





![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-2/misclassifed-bird-drone.jpg)







*********************************************************************************************

Execution Logs:-


*********************************************************************************************


 ----------------------------------------------------------------
EPOCH: 0 LR: 0.0002000000000000001 
Loss=0.7498657703399658 Batch_id=428 le=0.0006506348378616084 Accuracy=74.46: 100%|██| 429/429 [03:15<00:00,  2.19it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0141, Accuracy: 4828/5877 (82.15%)

EPOCH: 1 LR: 0.0006506348378616084 
Loss=0.17783458530902863 Batch_id=428 le=0.0015512687801251931 Accuracy=83.45: 100%|█| 429/429 [02:37<00:00,  2.72it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0121, Accuracy: 4991/5877 (84.92%)

EPOCH: 2 LR: 0.0015512687801251931 
Loss=0.2425840198993683 Batch_id=428 le=0.0019999998137962957 Accuracy=85.73: 100%|██| 429/429 [02:37<00:00,  2.72it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0140, Accuracy: 4857/5877 (82.64%)

EPOCH: 3 LR: 0.0019999998137962957 
Loss=0.35503485798835754 Batch_id=428 le=0.001965768043149073 Accuracy=88.06: 100%|██| 429/429 [02:37<00:00,  2.73it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0104, Accuracy: 5151/5877 (87.65%)

EPOCH: 4 LR: 0.001965768043149073 
Loss=0.8392808437347412 Batch_id=428 le=0.001865721457851869 Accuracy=89.84: 100%|███| 429/429 [02:37<00:00,  2.72it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0109, Accuracy: 5110/5877 (86.95%)

EPOCH: 5 LR: 0.001865721457851869 
Loss=0.0883207619190216 Batch_id=428 le=0.0017066780673578874 Accuracy=91.49: 100%|██| 429/429 [02:38<00:00,  2.71it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0105, Accuracy: 5174/5877 (88.04%)

EPOCH: 6 LR: 0.0017066780673578874 
Loss=0.20369087159633636 Batch_id=428 le=0.0014994764158976638 Accuracy=92.65: 100%|█| 429/429 [02:41<00:00,  2.66it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0108, Accuracy: 5135/5877 (87.37%)

EPOCH: 7 LR: 0.0014994764158976638 
Loss=0.1522112786769867 Batch_id=428 le=0.0012582369536012932 Accuracy=93.91: 100%|██| 429/429 [03:31<00:00,  2.02it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0110, Accuracy: 5223/5877 (88.87%)

EPOCH: 8 LR: 0.0012582369536012932 
Loss=0.1781913936138153 Batch_id=428 le=0.0009993997511572124 Accuracy=95.06: 100%|██| 429/429 [04:16<00:00,  1.67it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0118, Accuracy: 5209/5877 (88.63%)

EPOCH: 9 LR: 0.0009993997511572124 
Loss=0.3481016755104065 Batch_id=428 le=0.0007406041361632838 Accuracy=95.86: 100%|██| 429/429 [04:24<00:00,  1.62it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0114, Accuracy: 5224/5877 (88.89%)

EPOCH: 10 LR: 0.0007406041361632838 
Loss=0.08326173573732376 Batch_id=428 le=0.000499486602101368 Accuracy=96.74: 100%|██| 429/429 [04:21<00:00,  1.64it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0115, Accuracy: 5232/5877 (89.03%)

EPOCH: 11 LR: 0.000499486602101368 
Loss=0.007248550653457642 Batch_id=428 le=0.00029247891045221995 Accuracy=97.23: 100%|█| 429/429 [04:28<00:00,  1.60it/
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0121, Accuracy: 5246/5877 (89.26%)

EPOCH: 12 LR: 0.00029247891045221995 
Loss=0.005194753408432007 Batch_id=428 le=0.00013368829330534377 Accuracy=97.67: 100%|█| 429/429 [04:23<00:00,  1.63it/
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0122, Accuracy: 5225/5877 (88.91%)

EPOCH: 13 LR: 0.00013368829330534377 
Loss=0.03461006283760071 Batch_id=428 le=3.393606880539646e-05 Accuracy=97.85: 100%|█| 429/429 [04:15<00:00,  1.68it/s]
  0%|                                                                                          | 0/429 [00:00<?, ?it/s]
Test set: Average loss: 0.0125, Accuracy: 5234/5877 (89.06%)

EPOCH: 14 LR: 3.393606880539646e-05 
Loss=0.014611005783081055 Batch_id=428 le=2.0186203704465716e-08 Accuracy=98.07: 100%|█| 429/429 [04:39<00:00,  1.54it/
Test set: Average loss: 0.0125, Accuracy: 5241/5877 (89.18%)