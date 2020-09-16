# -*- coding: utf-8 -*-
"""
Last modified on 31 July 2020

@author: Dr Sen Hu (UCD)

This file corresponds to the way Ireland 2016 census data set being filtered in the paper:
    Hu et al (2020), "A spatial machine learning model for analyzing customers' lapse behaviour in life insurance", Annals of Actuarial Science.
"""

#%%
import pandas as pd

#%%
"""
Load the census data 
Omitted data directory here, fill in your own directory for the data set
"""

df = pd.read_csv('SAPS2016_SA2017.csv')
df.shape
df.head()
df.columns

census_final = pd.DataFrame()

#%%
"""
In the census data: 
Theme 1:  Age, marrital status
Theme 2:  Migration, ethnicity, religion and foreign language
Theme 3:  Irish language
Theme 4:  Families 
Theme 5:  Private households
Theme 6:  Housing
Theme 7:  Communal establishment
Theme 8:  Principal status
Theme 9:  Social class, socio-economic groups
Theme 10: Education
Theme 11: Commuting   # possibly use more?!
Theme 12: Disability, carers, general health
Theme 13: Occupation  # did not use, as reflected in social class in Theme 9
Theme 14: Industry
Theme 15: Motor car, internet

More descriptions about census variables are availabble in "SAPS_2016_Glosssry" file. 

We summarized the raw 764 variables in the census data into 69 summary variables to characterize small areas. 
These 69 summary variables can be groups into 5 categories:
Demographic information, Household composition, Housing, Socio-economic information, Employment. 
"""

#%%
"""
* Demographic information *

contains age, nationality, ethnic group, language information. 
"""

# age
Age0_4 = 100*(df["T1_1AGE0T"]+df["T1_1AGE1T"]+df["T1_1AGE2T"]+df["T1_1AGE3T"]+df["T1_1AGE4T"])/df["T1_1AGETT"]
Age5_14 = 100*(df["T1_1AGE5T"]+df["T1_1AGE6T"]+df["T1_1AGE7T"]+df["T1_1AGE8T"]+df["T1_1AGE9T"]+df["T1_1AGE10T"]+df["T1_1AGE11T"]+df["T1_1AGE12T"]+df["T1_1AGE13T"]+df["T1_1AGE14T"])/df["T1_1AGETT"]
Age25_44 = 100*(df["T1_1AGE25_29T"]+df["T1_1AGE30_34T"]+df["T1_1AGE35_39T"]+df["T1_1AGE40_44T"])/df["T1_1AGETT"]
Age45_64 = 100*(df["T1_1AGE45_49T"]+df["T1_1AGE50_54T"]+df["T1_1AGE55_59T"]+df["T1_1AGE60_64T"])/df["T1_1AGETT"]
Age65over = 100*(df["T1_1AGE65_69T"]+df["T1_1AGE70_74T"]+df["T1_1AGE75_79T"]+df["T1_1AGE80_84T"]+df["T1_1AGEGE_85T"])/df["T1_1AGETT"]

# nationality
EU_National  = 100*(df["T2_1UKN"]+df["T2_1PLN"]+df["T2_1LTN"]+df["T2_1EUN"])/df["T2_1TN"]
ROW_National = 100*(df["T2_1RWN"])/df["T2_1TN"]
Born_outside_Ireland  = 100*(df["T2_1TBP"]-df["T2_1IEBP"])/df["T2_1TBP"]

Minority = 100*( df["T2_2BBI"]+df["T2_2AAI"]+df["T2_2OTH"])/df["T2_2T"]
English = 100*( df["T2_6NW"]+df["T2_6NAA"] ) / df["T2_6T"]

census_final["Age0_4"] = Age0_4
census_final["Age5_14"] = Age5_14
census_final["Age25_44"] = Age25_44
census_final["Age45_64"] = Age45_64
census_final["Age65over"] = Age65over
census_final["EU_National"] = EU_National
census_final["ROW_National"] = ROW_National
census_final["Born_outside_Ireland"] = Born_outside_Ireland
census_final["Minority"] = Minority
census_final["English"] = English

#%%
"""
* Household Composition *

contains marrital status, family stage type, household type
"""

# Marrital status
Separated = 100*(df["T1_2SEPT"]+df["T1_2DIVT"])/df["T1_2T"]
Married = 100*df["T1_2MART"] / df["T1_2T"]
Single = 100*(df["T1_2SGLT"]+df["T1_2WIDT"])/df["T1_2T"]

# Family (cycle) type
Dink = 100*df["T4_5PFF"] / df["T4_5TF"]
Pensioner = 100*df["T4_5RP"]/df["T4_5TP"]  # retired person
LoneParent = 100*(df["T4_3FOPMCT"] + df["T4_3FOPFCT"]) / df["T4_4TF"]
NonDependentKids = 100*df["T4_4AGE_GE20F"] / df["T4_4TF"]
EmptyNest = 100*df["T4_5ENF"] / df["T4_5TF"]
NoChildrenFam = 100*df["T4_2_NCT"] / df["T4_2_TCT"]

# Household
SinglePersonFam = 100*(df["T5_2_1PP"]) / df["T5_2_TP"]  
HouseShare = 100*(df["T5_1GETFU_H"]+df["T5_1NHR_H"]+df["T5_1GENP_H"]) / df["T5_1T_H"]

census_final["Separated"] = Separated
census_final["Married"] = Married
census_final["Single"] = Single
census_final["Dink"] = Dink
census_final["Pensioner"] = Pensioner
census_final["LoneParent"] = LoneParent
census_final["NonDependentKids"] = NonDependentKids
census_final["EmptyNest"] = EmptyNest
census_final["NoChildrenFam"] = NoChildrenFam
census_final["SinglePersonFam"] = SinglePersonFam
census_final["HouseShare"] = HouseShare

#%%
"""
* Housing *
"""

# Household by type of accommodation
House = 100*df["T6_1_HB_H"] / df["T6_1_TH"]  
Flats = 100*df["T6_1_FA_H"] / df["T6_1_TH"]
# Household by type of occupancy
RentPublic = 100*df["T6_3_RLAH"] / df["T6_3_TH"]
RentPrivate = 100*df["T6_3_RPLH"] / df["T6_3_TH"]
OwnedMortgage = 100*df["T6_3_OMLH"]/ df["T6_3_TH"]
Owned = 100*df["T6_3_OOH"] / df["T6_3_TH"] 
# Household by number of rooms
Rooms = (df["T6_4_1RH"]+2*df["T6_4_2RH"]+3*df["T6_4_3RH"]+4*df["T6_4_4RH"]+5*df["T6_4_5RH"]+6*df["T6_4_6RH"]+7*df["T6_4_7RH"]+8*df["T6_4_GE8RH"]) / df["T6_4_TH"]
PeopleRoom = df["T1_1AGETT"] / (df["T6_4_1RH"] + 2*df["T6_4_2RH"] + 3*df["T6_4_3RH"] + 4*df["T6_4_4RH"] + 5*df["T6_4_5RH"] + 6*df["T6_4_6RH"] + 7*df["T6_4_7RH"] + 8*df["T6_4_GE8RH"])
# Household by central heating
NoCenHeat = 100*df["T6_5_NCH"] / df["T6_5_T"]
# Household by sewerage facility
SepticTank = 100* df["T6_7_IST"] / df["T6_7_T"]

census_final["House"] = House
census_final["Flats"] = Flats
census_final["RentPublic"] = RentPublic
census_final["RentPrivate"] = RentPrivate
census_final["OwnedMortgage"] = OwnedMortgage
census_final["Owned"] = Owned
census_final["Rooms"] = Rooms
census_final["PeopleRoom"] = PeopleRoom
census_final["NoCenHeat"] = NoCenHeat
census_final["SepticTank"] = SepticTank

#%%
"""
* Socio-economic information *
"""

# Higher education to degree or higher
HE = 100*(df["T10_4_ODNDT"]+df["T10_4_HDPQT"]+df["T10_4_PDT"]+df["T10_4_DT"]) / df["T10_4_TT"] 
# HE field of study: STEM, HASS, Health+Medicine
HEstem = 100*( df["T10_3_SCIT"]+ df["T10_3_ENGT"] ) / df["T10_3_TT"]
HEhass = 100*( df["T10_3_ARTT"]+ df["T10_3_HUMT"] + df["T10_3_SOCT"] ) / df["T10_3_TT"]
HEhealth = 100*( df["T10_3_HEAT"]+ df["T10_3_AGRT"] ) / df["T10_3_TT"]
# Principle status: employment
Employed = 100*df["T8_1_WT"] / df["T8_1_TT"]
# Number of cars: more than two cars
TwoCars = 100*(df["T15_1_2C"] + df["T15_1_3C"] + df["T15_1_GE4C"]) / (df["T15_1_NC"]+df["T15_1_1C"]+df["T15_1_2C"]+df["T15_1_3C"]+df["T15_1_GE4C"])
# Jouney to work/college: public transport
JTWpublic = 100*(df["T11_1_BUT"] + df["T11_1_TDLT"]) / df["T11_1_TT"]
JTWcar = 100*(df["T11_1_CDT"] + df["T11_1_CPT"]) / df["T11_1_TT"]
JTWvan = 100*df["T11_1_VT"] / df["T11_1_TT"]
JTWhome = 100*df["T11_1_WMFHT"] / df["T11_1_TT"]
# Health: bad and very bad general health status 
Health = 100*(df["T12_3_BT"] + df["T12_3_VBT"]) / df["T12_3_TT"]

# Social class (on the basis of occupation)
SC_professional = 100*df["T9_1_PWT"] / df["T9_1_TT"]
SC_managerial = 100*df["T9_1_MTT"] / df["T9_1_TT"]
SC_nonmanual = 100*df["T9_1_NMT"] / df["T9_1_TT"]
SC_skilled = 100*df["T9_1_ST"] / df["T9_1_TT"]
SC_semi = 100*df["T9_1_SST"] / df["T9_1_TT"]
SC_unskilled = 100*df["T9_1_UST"] / df["T9_1_TT"]

# Socio-economic group (on the basis of skills and education)
Employer = 100*df["T9_2_PA"] / df["T9_2_PT"]
HighProfessional = 100*df["T9_2_PB"] / df["T9_2_PT"]
LowProfessional = 100*df["T9_2_PC"] / df["T9_2_PT"]
Nonmanual = 100*df["T9_2_PD"] / df["T9_2_PT"]
Skilled = 100*df["T9_2_PE"] / df["T9_2_PT"]
Semiskilled = 100*df["T9_2_PF"] / df["T9_2_PT"]
Unskilled = 100*df["T9_2_PG"] / df["T9_2_PT"]
HomeWork = 100*df["T9_2_PH"] / df["T9_2_PT"]
Farmer = 100*df["T9_2_PI"] / df["T9_2_PT"]

census_final["HE"] = HE
census_final["HEstem"] = HEstem
census_final["HEhass"] = HEhass
census_final["HEhealth"] = HEhealth
census_final["Employed"] = Employed
census_final["TwoCars"] = TwoCars
census_final["JTWpublic"] = JTWpublic
census_final["JTWcar"] = JTWcar
census_final["JTWvan"] = JTWvan
census_final["JTWhome"] = JTWhome
census_final["Health"] = Health
census_final["SC_professional"] = SC_professional
census_final["SC_managerial"] = SC_managerial
census_final["SC_nonmanual"] = SC_nonmanual
census_final["SC_skilled"] = SC_skilled
census_final["SC_semi"] = SC_semi
census_final["SC_unskilled"] = SC_unskilled
census_final["Employer"] = Employer
census_final["HighProfessional"] = HighProfessional
census_final["LowProfessional"] = LowProfessional
census_final["Nonmanual"] = Nonmanual
census_final["Skilled"] = Skilled
census_final["Semiskilled"] = Semiskilled
census_final["Unskilled"] = Unskilled
census_final["HomeWork"] = HomeWork
census_final["Farmer"] = Farmer

#%%
"""
* Employment *
"""

Students = 100*df["T8_1_ST"] / df["T8_1_TT"]
Unemployed = 100*(df["T8_1_ULGUPJT"]+df["T8_1_LFFJT"]) / df["T8_1_TT"]
EconInactFam = 100*df["T8_1_LAHFT"] / df["T8_1_TT"]

# Occupation industry
Agric = 100*df["T14_1_AFFT"] / df["T14_1_TT"]
Construction = 100*df["T14_1_BCT"] / df["T14_1_TT"]
Manufacturing = 100*df["T14_1_MIT"] / df["T14_1_TT"]
Commerce = 100* df["T14_1_CTT"] / df["T14_1_TT"]
Transport = 100*df["T14_1_TCT"] / df["T14_1_TT"]
Public = 100*df["T14_1_PAT"] / df["T14_1_TT"]
Professional = 100*df["T14_1_PST"] / df["T14_1_TT"]

census_final["Students"] = Students
census_final["Unemployed"] = Unemployed
census_final["EconInactFam"] = EconInactFam
census_final["Agric"] = Agric
census_final["Construction"] = Construction
census_final["Manufacturing"] = Manufacturing
census_final["Commerce"] = Commerce
census_final["Transport"] = Transport
census_final["Public"] = Public
census_final["Professional"] = Professional

#%%
"""
* Misc *
"""

# Internet connected HH with Broadband
Broadband = 100*df["T15_3_B"] / (df["T15_3_B"] + df["T15_3_OTH"])     
# Households with Internet       
Internet = 100 * (df["T15_3_B"] + df["T15_3_OTH"]) / df["T15_3_T"]            

census_final["Broadband"] = Broadband
census_final["Internet"] = Internet

#%%
"""
Explore the summarized variables
Add the small area variable
"""

census_final.shape
census_final.head()
census_final.columns

census_final["SAID"] = df.iloc[:,1]
census_final.head(20)

#%%
"""
The small area variable in the census has the prelix "SA2007"
Need to eliminate it, only keep small area ID number. 
"""

def myconvertor(cha):
    return cha[7:]
census_final["SAID"] = list(map(myconvertor, df.iloc[:,1]))

census_final.head()

#%%
"""
Add county name into the summarized census data set
so that we can extract Dublin region census data

We use the "Small_Areas_Boundaries_2015" file from the census website
which contains all descriptive geo information about each small area

Note to fill in your own data directory
"""

refkey = pd.read_csv('Small_Areas_Boundaries_2015.csv')
refkey.head()
refkey.shape
refkey.columns
refkey = refkey[["COUNTYNAME","SMALL_AREA"]]

Ireland = pd.merge(census_final, refkey, left_on='SAID', right_on='SMALL_AREA', how='left', sort=False)
Ireland.head()
Ireland.shape

# save the data
#Ireland.to_csv("NewCensusData_final_Ireland.csv")

Ireland["COUNTYNAME"].value_counts()
Dublin = Ireland[Ireland['COUNTYNAME'].isin(['Fingal','Dublin City','South Dublin', 'Dún Laoghaire-Rathdown'])]
Dublin.shape
Dublin.head()
Dublin["COUNTYNAME"].value_counts()

Dublin["SAID"].isnull().sum()

# save the data
#Dublin.to_csv("NewCensusData_final_Dublin.csv")
