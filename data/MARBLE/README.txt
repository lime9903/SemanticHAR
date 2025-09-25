The dataset is organized in different scenarios. Each scenario represents a sequence of activities designed for
each subject involved in the data collection process related to the scenario itself.
Each scenario has been performed different times, so we have different instances related to the same scenario.
Each instance of the same scenario has been performed by a different combination of subjects. So the instances
will differ based on the way in which the activities have been executed by the subjects.
Data related to each instance of each scenario are separated based on the subject they have been collected from.



Dataset Structure:
dataset 
   |__ scenario-id
          |__ instance-number
		 |__ environmental.csv
                 |__ subject-id
                       |__ accelerometer.csv
                       |__ magnetometer.csv
                       |__ gyroscope.csv
                       |__ barometer.csv
                       |__ smartphone.csv
                       |__ locations.csv (user's locations ground truths, based on time intervals)
                       |__ labels.csv (user's activities ground truths, based on time intervals)



Scenarios details:
Each scenario is identified by an id. The id is composed of 
	- a letter (A, B, C, D)
	- the number of subjects involved in the scenario (1, 2, 4)
	- the period of the day simulated in the scenario (morning (m), afternoon (a), evening (e)). When you find "mae", 
	this means that a whole day has been simulated
For example, the scenario A2m indicates that it has been performed by two subjects, simulating the morning of an arbitrary day



CSV details:
All the timestamps are UNIX timestamps expressed in milliseconds
- environmental.csv contains the environmental events triggered by all the subjects that partecipated
	to the acquisition of the instance of a scenario.
	A detailed description of the environmental sensors is provided later
- accelerometer.csv contains the data collected from the accelerometer installed in the smartwatch worn by the subject.
	Each row of the csv will contain the data related to the x, y, and z axes of the sensor (columns x, y, z), and 
	the timestamp related to the acquisition (column ts)
- magnetometer.csv contains the data collected from the magnetometer installed in the smartwatch worn by the subject.
	Each row of the csv will contain the data related to the x, y, and z axes of the sensor (columns x, y, z), and 
	the timestamp related to the acquisition (column ts)
- gyroscope.csv contains the data collected from the gyroscope installed in the smartwatch worn by the subject.
	Each row of the csv will contain the data related to the x, y, and z axes of the sensor (columns x, y, z), and 
	the timestamp related to the acquisition (column ts)
- barometer.csv contains the data collected from the barometer installed in the smartwatch worn by the subject.
	Each row of the csv will contain the value (column value), and the timestamp related to the acquisition (column ts)
- smartphone.csv contains the smartphone events triggered by the subject during the acquisition.
- locations.csv contains the information regarding the smart-home locations in which the subject performed his/her activities during the acquisition.
	This information is provided based on time intervals
- labels.csv contains the information regarding the activities performed by the subject during the acquisition.
	This information is provided based on time intervals



Environmental Sensors details:
Sensor ID 		Sensor Type 		Description
R1 			Magnetic 		pantry
R2 			Magnetic 		cutlery drawer
R5 			Magnetic 		pots drawer
R6 			Magnetic 		medicines cabinet
R7 			Magnetic 		fridge
E1 			Smart Plug 		stove
E2 			Smart Plug 		television
P1 			Pressure Mat 		dining room chair
P2 			Pressure Mat 		office chair
P3 			Pressure Mat 		living room couch
P4 			Pressure Mat 		dining room chair
P5 			Pressure Mat 		dining room chair
P6 			Pressure Mat 		dining room chair
P7 			Pressure Mat 		living room couch
P8 			Pressure Mat 		living room couch
P9 			Pressure Mat 		living room couch


Smartphone Events details:
S1 			Smartphone 		make a phone call
S2 			Smartphone 		answer to an incoming call



Dataset Activities:
1  Answering Phone: 	the actor extracts a smartphone from the pocket and answers to an incoming call
2  Clearing Table: 	the actor collects dishes, glasses and cutlery, put the bottles in the fridge, and 
				remove the tablecloth from the table
3  Cooking: 		this activity simulates the cooking of milk or pasta. For milk, the actor takes
				a pot, fills it with some milk taken from the fridge, and warm it by using the cooker.
				For pasta, the actor takes a pot, fills it with some tap water, and warm it by using the cooker. Then
				he/she adds salt and pasta to the pot. After a while, he/she drains pasta and, if he/she wants to, dresses 
				it with condiments taken from the fridge. The order of these action is left to the discretion of each actor
4  Eating*: 		the actor sits on a dining room chair and starts eating and drinking
5  Entering Home:	the actor enters in the smart-home and, if necessary, takes off his/her coat, hanging it on the coat hanger
6  Leaving Home: 	the actor wears the coat, if necessary, and leaves the smart-home
7  Making Phone Call: 	the actor extracts the smartphone from the pocket, types a phone number, and makes a call
8  Preparing Cold Meal: the actor prepares a sandwich or a salad. For the sandwich, he/she takes the bread, cuts it with a knife,
				and fills it with food taken from the fridge.
				For the salad, he/she takes the salad from the fridge, puts it in a bowl, and mixes it with salt and oil
9  Setting Up Table: 	the actor puts tablecloth, dishes, glasses, napkins, cutlery, and beverages on the table
10 Taking Medicines: 	the actor takes a medicine from the proper drawer and gets it; he/she can sit on one of the smart-home chairs, 
				and, if necessary, drinking a glass of water taken from the kitchen
11 Using Pc: 		the actor sits on the office chair and uses the pc
12 Washing Dishes: 	the actor washes the dishes in the sink
13 Watching Tv: 	the actor switches on the television if is off and sits on a living room chair; at the end of the activity,
				he/she can switch off the television if nobody else is watching it

*Note that the Eating label is used also when a subject just drink a glass of water.

In the labels.csv files you will even find information about the TRANSITION between an activity to the subsequent one performed by the subject

