 
_________________________________________________________________
                   S E N S I C H I P S                           
-----------------------------------------------------------------
         l e a r n i n g   m i c r o s e n s o r s     
_________________________________________________________________
 
                - SLM-Studio, Ver 1.2.7 -              
 
 
Date: 13-04(April)-2025
 
Experiment starts at: 09:13:10.328
 
-----------------------------------------------------------------
 
Country Name: Italy
Region Name: 
City Name: 

Computer Name:  LAPTOP-QCLA7KG1
Logged User:  jampy
 
Operating system: Windows 11
 
Processor Name: Intel(R) Core(TM) i3-1005G1 CPU @ 1.20GHz
Microarchitecture: Ice Lake (Client)
Frequency: 1.19 GHz
 
Total memory: 8.4 GB
Available memory: 2.7 GB
 
Installation folder: C:\Sensichips\SLM-Studio

_________________________________________________________________
Optional notes: 



_________________________________________________________________


                    SLM-Studio Settings 

Configuration:        [RUN8][MANUAL]
Driver:               [WINUX_COMUSB]
Serial port:          [COM3][MANUAL]
Protocol:             [SENSIBUS][No Address]
Cluster ID:           [0X08]
Reference Voltage:    [RATIOMETRIC]
Measurement folder:   [./measures]
Controller:           [ESP8266]
Baud rate:            [460800]
API owner:            [PC]
Bandgap operation:    [NONE]
Host controller:      [PC]

_________________________________________________________________


                    Experiment Batch 

Experiment File Name: CheckAir.xml

-----------------------------------------------------------------
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sensichips>
  <experiment delay="0" globalRepetition="100000" 
	id="CheckAir" outputDecimation="1" >

<!--200HZ(IN-PHASE,QUADRATURE) PER OGNI TEMP -->
	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="150"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="200"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="250"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		 OFFCHIP_SENSIMOX_SnAu
	</sensor> 

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="300"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="350"
	SENSIMOX_Frequency="200"SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="400"
	SENSIMOX_Frequency="200"SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>


	  <!-- 78kHZ(SINGLE) , 200HZ(IN-PHASE,QUADRATURE) PER OGNI TEMP -->
	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="150"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="200"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="250"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor> 

	<sensor save="ALL" plot="IN-PHASE"
	 autorange="false" SETwaitGET = "0" filter = "1"
	 SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="300"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="350"
	SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="400"
	SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>


	  <!-- 312.5kHZ(IN-PHASE,QUADRATURE) PER OGNI OXIDE -->
	<sensor save="ALL" plot="IN-PHASE,QUADRATURE" autoscale="false" filter="1"
	autorange="false" >
		ONCHIP_ALUMINUM_OXIDE_OUT4
	</sensor>

	<sensor save="ALL" plot="IN-PHASE,QUADRATURE" autoscale="false" filter="1"
	autorange="false" >
		ONCHIP_ALUMINUM_OXIDE_OUT7
	</sensor> 

 </experiment>
</sensichips>


-----------------------------------------------------------------
_________________________________________________________________


_________________________________________________________________

       Experiment Batch (showing all default parameters)

-----------------------------------------------------------------
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sensichips>
  <experiment fillBufferBeforeStart="false" delay="0" globalRepetition="100000" 
	id="CheckAir" outputDecimation="1" >


	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="150"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="200"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="250"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		 OFFCHIP_SENSIMOX_SnAu
	</sensor> 

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="300"
	SENSIMOX_Frequency="200" SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="350"
	SENSIMOX_Frequency="200"SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="400"
	SENSIMOX_Frequency="200"SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>


	  
	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="150"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="200"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="250"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor> 

	<sensor save="ALL" plot="IN-PHASE"
	 autorange="false" SETwaitGET = "0" filter = "1"
	 SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="300"
	SENSIMOX_Rsense="5000">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="350"
	SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>

	<sensor save="ALL" plot="IN-PHASE"
	autorange="false" SETwaitGET = "0" filter = "1"
	SENSIMOX_Mode="NOTUSED"  SENSIMOX_Temperature="400"
	SENSIMOX_Rsense="500">
		OFFCHIP_SENSIMOX_SnAu
	</sensor>


	  
	<sensor save="ALL" plot="IN-PHASE,QUADRATURE" autoscale="false" filter="1"
	autorange="false" portLabel="PORT5" portValue="00101" rsense="50000" inGain="50" outGain="4" contacts="TWO" frequency="78125.0" Harmonic="FIRST_HARMONIC" DCBiasP="0" DCBiasN="0" modeVI="VOUT_IIN" measureTechnique="EIS" measureType="CONDUCTANCE" filter="1" phaseShiftMode="Quadrants" phaseShift="0" iq="IN_PHASE" conversionRate="500" NData="1" inPortADC="IA" sequentialMode="false" burstMode="false" fillBufferBeforeStart="false" >
		ONCHIP_ALUMINUM_OXIDE_OUT4
	</sensor>

	<sensor save="ALL" plot="IN-PHASE,QUADRATURE" autoscale="false" filter="1"
	autorange="false" portLabel="PORT5" portValue="00101" rsense="50000" inGain="50" outGain="7" contacts="TWO" frequency="78125.0" Harmonic="FIRST_HARMONIC" DCBiasP="0" DCBiasN="0" modeVI="VOUT_IIN" measureTechnique="EIS" measureType="CONDUCTANCE" filter="1" phaseShiftMode="Quadrants" phaseShift="0" iq="IN_PHASE" conversionRate="500" NData="1" inPortADC="IA" sequentialMode="false" burstMode="false" fillBufferBeforeStart="false" >
		ONCHIP_ALUMINUM_OXIDE_OUT7
	</sensor> 

 </experiment>
</sensichips>


-----------------------------------------------------------------
_________________________________________________________________

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
_________________________________________________________________
                          Debug Info:                            
_________________________________________________________________
 
-----------------------------------------------------------------
 
Port name: Dispositivo seriale USB (COM3)
Port description: USB JTAG/serial debug unit
Port location: 0-0.6
Port path: \\.\COM3
System port name: COM3
 
Jar file path: 
 
/C:/Sensichips/SLM-Studio/SensiplusWinux.jar
 
-----------------------------------------------------------------

