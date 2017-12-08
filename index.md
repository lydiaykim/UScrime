---
title: Predicting and Preventing Murder in the U.S.
---



## Research questions
1. What are the most important predictors of reductions in murder rates in Metropolitan areas of the US between 2006 and 2016? 
1. From among these, which are the most important policy measures that law enforcement officials can adopt to reduce murder rates? 


## Background
The goal of this project is to shed light on the areas of the country experiencing the highest rates of murder and violence. 

Overall, murder rates in the US rose in 2016 and 2017 for the first time since 2006 (Williams 2017). President Trump has made addressing rising crime and murder rates as a key policy priority in his administration. Pledging that his administration would put a stop to this ?American carnage? (White House, 2017). 

Popular and policy journalists writing at the time of the inauguration pointed out that he crime figures Trump was raising alarm bells about may have been misleading. The overall rise in crime was in fact driven by a few outliers among the metropolitan areas of the country with increasing rates of violence such as Baltimore, Chicago, and Washington DC (McCray, 2017). 

Identifying the causes of changes ? whether positive reductions in homicides or the more troubling increases we have seen in some areas in the last two years ? is a key policy question for law enforcement officials. Our project will use a variety of publicly available data sources to identify predictors of increases and decreases in crime rates across metropolitan areas in the US.  

Since crime has been falling in most areas over the last decade, we will focus on identifying predictors of decreases in the murder rate, rather than the predictors of the outlier increases of the last two years.  


## Data

### Census DataWe use census data from 2006-2016. This data englobes and includes estimates of the population?s characteristics. We use a subdivision of Metropolitan Statistical Areas of the US, as this is the relevant unit for the study. For the purpose of this project we selected a subgroup of variables that might be relevant to the crime level of the region. See the next section for our method for choosing predictors. 

### Crime DataWe scraped crime data from the FBI for each of the MSAs. The data we used to create our database are:- Total Reported Murders: number of murders that were reported by the MSA (might have not been 100% reporting)- Estimated Murders: estimate variable of the total number of murders for MSAs that had less than 100% reporting- Murder Rate: number of murders for every 100,000 inhabitants

In this project, we focus mainly on the murder rate. 

### Other DataWe use external data sources on access to firearms and presence of firearms. Evidence suggests that areas with more firearms have higher rates of homicide (Hepburn et al 2004). We thus use large datasets to determine if MSAs with tighter laws regarding gun access and lower prevalence of gun ownership have lower homicide rates. We use two datasets to do this: - Firearms Provisions in US States (Bouchet, 2017)- Federal Firearm Licenses (ATF, 2017 via Bilgour et al 2017)




