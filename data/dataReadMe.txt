U	Y	REF

File		Info
sineAndJump	1000,000 data points of sinusoidal function plus jumps. The frequency 			of the sine part is an integer from 1 to 100 Hz. The jumps are the 			same as "randomJump.txt". The amplitude of the jumps vary from 0.2 to 			3. The amplitude of the sine wave is 3. Noise to signal ratio = 10. 			Essentially, I element-wise added the jump and sine references.

bigSineFile	1000,000 data points of sinusoidal functions. Frequency is a integer 			between 1 and 100 Hz. Noise added with Noise-to-Signal ratio 			=10. Amplitude = 1 not including noise, with noise it goes to about 1.5

randomJump	600,000 data points of random jumps to a set point. Value of set point 			is between -0.2 and -3 and 0.2 and 3. The jumps last for 10 to 100 			time steps. No noise added

manySinesWithRef	Same as bigSineFile.txt but 10,000 data points frequency 				ranges from 10Hz to 100Hz in steps of 10. Amplitude = 1.

manySines	Same as manySinesWithRef but only contains [u y] data, not [u y ref]

testFile	File for testing the model, contains 1000 data points. This is a sine 			wave that is perturbed a little bit.

newMPCData	Not sure...
