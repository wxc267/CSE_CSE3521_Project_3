Project:    CSE 3521 Project 3
Date:       Apr 27, 2016
Group:      #8
Members:    Chuhan Feng, Colin Dolan, Xiaochi Weng

Files:      learning_agent.py .............. Learning algorithm
		    logistic_regression.py ......... Core implementation
		    TrainLogReg.py ................. Project part 1
		    TestLogReg.py .................. Project part 2
		    Accuracy.py..................... Project part 3

Execution:  python3.5 TrainLogReg.py trainingFeatureFile trainingLabelFile modelFile D Niter 
		    python3.5 TestLogReg.py modelFile testFeatureFile predLabelFile D
			python3.5 Accuracy.py predLabelFile trueLabelFile

Note:       All .dat files are located in data folder. When specifying file names, no
            need to add 'data/' before them.