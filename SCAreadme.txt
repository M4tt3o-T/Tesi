_________________________________________________________________
                    S E N S I C H I P S   
-----------------------------------------------------------------
         l e a r n i n g   m i c r o s e n s o r s     
_________________________________________________________________
 
		   SCA air sensor dataset

Date: 04-02(February)-2025
 
-----------------------------------------------------------------
 
This data set has been acquired with the SCA air sensor, files are saved in the .csv format.
Data includes 16 impedance measurements at AC frequencies of 78125Hz and 200Hz acquired from two different gas sensors: a Metal Oxide hotplate sensor functionalized with a sensitive SnO2+Au film and from an interdigit Al2O3 oxidized aluminum sensor. 

- 12 acquisitions are performed from the SnO2+Au sensor heated from 150°C to 400°C with six temperature steps of 50°C each. For each temperature the IN-PHASE component is acquired at 78125Hz and 200Hz. This technique is referred as the Temperature Cycled Operation (TCO) with AC impedance readout.
- 4 acquisitions are performed from the Al2O3 sensor which is stimulated at two different high and low Voltages and read at 78125Hz only, both the IN-PHASE and QUADRATURE components are acquired. This technique is referred as the Voltage Cycled Operation (VCO) with AC impedance readout.

The TRAINING directory includes measurements to be used for the development of machine learning models and its validation. 
Experiments were performed by adding each compound in liquid form inside a glass and let it slowly evaporate in an enclosed environment, alternatively in open air exposed to environmental turbulence.
The slow evaporation with no turbulence is to allow the sensor to acquire minimum 200 samples from low to higher concentrations, so that the model can be trained to classify at different concentrations.
10 or more such experiments were repeated for each of the chemicals.

- The RAW subdirectory includes data as it is acquired from the sensor with its built-in 16-bits Analog to Digital converter, represented with a complement two integer number ranging from -32768 to 32767. The 16 raw sensor acquisitions form the columns inside the .csv files of this directory.

- The NORMALIZED subdirectory includes the same data as the RAW directory having applied a simple mathematical difference from the actual sample to the air environment measurement used as reference baseline. 

The TEST subdirectory includes measurements performed in random conditions and inducing air turbulence to better approximate real world operation.
These experiments can be used to test the models and improve them if needed.




