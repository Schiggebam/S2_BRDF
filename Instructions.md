## BRDF large scale analysis

The analysis is carried out on a HPC platform connected to the Sentienl-2 L2A archive.   
Objective: Understand BRDF effects in Sentinel-2 better over different landcover types  
Data: In 2025 Sentinel-2A and 2B flew in formation and their swaths partially overlapped. That means, many pixels have been scanned from different view angles twice only minutes appart. 

The S2 data is organized like this: DIR\_BASE\_L2A {grid\_id\[:2\]}/{grid\_id\[2:3\]}/{grid\_id\[3:}/{year}/{month (e.g. „04“)}/{scene\_id}.SAFE/GRANUALE/{scene\_id}/IMG\_DATA/R10m\|R20m/\*\_{Band (e.g. B04) B04\_10m.jp2

Sentienl-2 swaths are wider than a tile. So one should always look at to horizonally adjacent tiles to have the the entire overlaping swath region covered. 

Generally i always want a near-Nadir (meaning less than 4.0 degrees off-nadir) comparison vs a off-Nadir reflectance comparison

I want to compare the effects in various landcover types (e.g. forests, permanent grassland, cropland), in various conditions of the landcover type (e.g. cropland what is the NDVI, is it bare cropland or not) and in different seasons perhaps

For the landcover i currently use WorldCOver 2021\. It is outdated but i still use it

There is also a floating idea to use L1C data and do the atmos correction using Py6s to avoid that Sen2cor correction based on different tiles (and different paramters for the correction) interferes with the results too much. 

Generally, only look at pairs with little cloud cover and with sun zenith angle smaller than 70 degree

In the respository you find code for much of this already  
You also find a separate code for finding the S2A and S2B pairs.  
I will provide a list of grid ids (roughly 600\) in which i search for pairs and perfrom the analyis. 

Your task is to organize the code into a standalone program that can be exectued with a slurm batch script (an example is provided) for a selected period of time. I will run the script so that each job processes the entire tile-list but only for a month or less for instance. 

It is specfically important that the program understands that if i have a hit for the same day in 31UDP and 31UCP (horizonatlly adjacent tiles) that thsi is the same overpass and in each in find either the near-nadir S2A or near-nadir S2B configuration.  
In this case, scene-ids would look like this: 

S2A\_MSIL2A\_20250409T105701\_N0511\_R094\_T31UDP\_20250409T173716  
S2B\_MSIL2A\_20250409T104619\_N0511\_R051\_T31UDP\_20250409T120924  
S2A\_MSIL2A\_20250409T105701\_N0511\_R094\_T31UCP\_20250409T173716  
S2B\_MSIL2A\_20250409T104619\_N0511\_R051\_T31UCP\_20250409T120924

and this is a pair or 2 S2A-S2B matchups that would be analyzied together. They can then give information about the forward and backward scatter domain.

In general be aware of the position of the sun, the BRDF geometry.

When reading S2 data, apply the reduction by \-1000 (as written in the metadata) and mask it with the SCL layer (keep only SCL in \[3,4,5\])

Store a comprensive output, data of all sampled locations in a database or otherwise sutable for further processing. Make sure that it is clear from which scene the data came and when.



Provide the base of the SLURM script, i will put in the header with the server specific configuration
The tile_list will look list this: 
- 32UQU
- 32UPU
- ...
(it essentially defines the region of interest)


I define a output directory. Each process should write in a own subdirectory with a unique name that relates to the data. Make sure to make comprehensive outputs. If a process fails, make sure to check if the data is already there before recomputing it.

Put the selection criteria into a yaml file: 
Scenes should have less than 20% cloud cover (all 4 scenes e.g. the 2 adjacent pairs of S2A and S2B)

Before implementing, do you need more information, what should we consider?

