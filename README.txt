#########################################################################
Subject:    ECE595Z (Digital System Design Automation)
Project:    Term Project - Spring 2020
Authors:    Badrinarayanan Nandhakumar & Venkatesh Bharadwaj Srinivasan
Code:       Machine Learning based SAT Solver
#########################################################################

This folder contains the source code for Machine Learning variable branching heuristic
for Minisat Solver in C++. 

================================================================================
DIRECTORY OVERVIEW (Courtesy: MiniSat):

mtl/            Mini Template Library related to the implementation of data structures
utils/          Generic helper code related to system level functions (I/O, Parsing, CPU-time, etc)
core/           A core version of the solver (We will be using this for our implementation)
	Solver.cc - 
Makefile		This makefile, when compiled, links itself to the template.mk in the ./mtl folder and
				Makefile in the ./core folder
README

Since we are primarily using core/, we will see the important files present in /core
	1. Solver.cc and Solver.h - The code for implementing the solver functionalities and to initialize the variables used, functions respectively
	2. SolverTypes.h - Assigns the branching heuristics and the other techniques like LBD, Rapid deletion that can be implemented in the SAT Solver
	3. Main.cc - The main function of the core code
	4. Dimacs.h - Converting to DIMACS
Steps to run the code:
a. Requirements:
    1. gcc compiler
    2. Please run this code from qstruct.ecn.purdue.edu

b. Steps to run the code
    1. From the root folder, type 'make' in the Terminal window. 
    2. Once the build is complete, go inside the /core folder.  
    3. If you find the mySAT application, you can run the application on any benchmark. You need to perform this step in order to ignore ./ at the start. Kindly type in the terminal window, 
    		export PATH=$PATH:/home/venkatesh/Desktop/sgd/core (The application is here in my case. You can replace with your path)
 		Then use the following command:
        	mySAT <benchmark.cnf> output.txt
    4. The output for the code will be displayed on the screen and also be written onto the file 'output.txt'. 

Data structures:
We make use of 2 hyperparameters namely, Global Learning Rate and Literal Block Distance. We tend to maximize the GLR using ML heuristic technique of Stochastic Gradient Descent. The LBD, when small, indicates the clause to possess good quality. We use a parameter called br_heuristic which, when passed as an argument in the SolverTypes.h executes the corresponding branching heuristic. 

Techniques implemented apart from the already existing VSIDS:
1. Greedy VSIDS
2. Stochastic Gradient Descent 

The code has been commented very well in most of the lines of my self-implementation in the MiniSat code. Hence, the comments are lucid for people to understand. The logic / algo mentioned in the publication 'Machine Learning For SAT Solvers' was very helpful in coding the algorithm line by line. 
