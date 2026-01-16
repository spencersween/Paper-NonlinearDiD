/*******************************************************************************
*                                                                              *
*   NONLINEAR DIFFERENCE-IN-DIFFERENCES PROJECT                                *
*   Data Preparation and Panel Construction                                    *
*                                                                              *
*   Author:  Spencer Sween                                                     *
*   Purpose: Merge and clean county-level entrepreneurship data with           *
*            state-level tax incentive policies for DiD analysis               *
*                                                                              *
*   Input Files:                                                               *
*     - pdit.csv (Panel Database of Incentives and Taxes)                      *
*     - Entrepreneurship_by_County_academic.dta (Startup Cartography Project)  *
*     - historical_county_populations_v2.csv                                   *
*     - allhlcn90.xlsx (QCEW 1990 county-level data)                           *
*                                                                              *
*   Output Files:                                                              *
*     - scp_pdit_county.dta (Final analysis dataset)                           *
*                                                                              *
*******************************************************************************/

clear all
set more off
cls


/*******************************************************************************
*                                                                              *
*   SECTION 0: DEFINE FILE PATHS                                               *
*                                                                              *
*******************************************************************************/

cd "/Users/spencersween/Dropbox/Paper -- Nonlinear DiD/"

global raw          "data/raw"
global intermediate "data/intermediate"
global final        "data/final"


/*******************************************************************************
*                                                                              *
*   SECTION 1: CLEAN PANEL DATABASE OF INCENTIVES AND TAXES (PDIT)             *
*                                                                              *
*   This section processes state-level tax incentive data, including:          *
*     - R&D Tax Credits                                                        *
*     - Investment Tax Credits                                                 *
*   Creates treatment cohort indicators for staggered DiD design.              *
*                                                                              *
*******************************************************************************/

import delimited "${raw}/pdit.csv", clear varn(1)

*-------------------------------------------------------------------------------
* Filter and Rename Variables
*-------------------------------------------------------------------------------

* Keep 'All Export' industries (consistent with published paper methodology)
keep if industry == "All Export"

* Select and rename relevant variables
keep state baseyear researchanddevelopmentcredit investmenttaxcredit
rename state state_abr
rename baseyear year
rename researchanddevelopmentcredit rnd
rename investmenttaxcredit itc

*-------------------------------------------------------------------------------
* Create Treatment Cohort Indicators (Staggered Adoption Design)
*-------------------------------------------------------------------------------

* R&D Tax Credit adoption cohort (G_rnd)
* First year with positive R&D credit; 0 = never treated
gen temp_rnd = year if rnd > 0
hashsort state_abr year
by state_abr: gegen G_rnd = min(temp_rnd)
replace G_rnd = 0 if missing(G_rnd)
drop temp_rnd

* Investment Tax Credit adoption cohort (G_itc)
* First year with positive ITC; 0 = never treated
gen temp_itc = year if itc > 0
hashsort state_abr year
by state_abr: gegen G_itc = min(temp_itc)
replace G_itc = 0 if missing(G_itc)
drop temp_itc

* Combined treatment cohort (G_both)
* Earliest adoption of either credit
gen G_both = G_rnd
replace G_both = G_itc if G_itc > 0 & G_rnd > 0 & G_itc < G_rnd 
replace G_both = G_itc if G_itc > 0 & G_rnd == 0

*-------------------------------------------------------------------------------
* Create Treatment Indicators
*-------------------------------------------------------------------------------

* Ever-treated indicators
gen treat_rnd = G_rnd != 0
gen treat_itc = G_itc != 0
gen treat_both = (treat_rnd | treat_itc)

* Treat-post indicators (treatment status in each year)
gen treat_post_rnd = year >= G_rnd & G_rnd != 0
gen treat_post_itc = year >= G_itc & G_itc != 0
gen treat_post_both = year >= G_both & G_both != 0

*-------------------------------------------------------------------------------
* Finalize and Save
*-------------------------------------------------------------------------------

keep state_abr year rnd itc G_rnd G_itc G_both ///
	treat_rnd treat_itc treat_both ///
	treat_post_rnd treat_post_itc treat_post_both
	
order state_abr year rnd itc G_rnd G_itc G_both ///
	treat_rnd treat_itc treat_both ///
	treat_post_rnd treat_post_itc treat_post_both

hashsort state_abr year

* Clear variable labels
foreach v of varlist * {
	label var `v' ""
}

save "${intermediate}/pdit.dta", replace


/*******************************************************************************
*                                                                              *
*   SECTION 2: CLEAN STARTUP CARTOGRAPHY PROJECT (COUNTY-LEVEL)                *
*                                                                              *
*   This section processes county-level entrepreneurship metrics:              *
*     - SFR: Startup Formation Rate                                            *
*     - EQI: Entrepreneurial Quality Index                                     *
*     - Growth: High-growth firm indicator                                     *
*                                                                              *
*******************************************************************************/

use "${raw}/Entrepreneurship_by_County_academic.dta", clear

*-------------------------------------------------------------------------------
* Initial Variable Processing
*-------------------------------------------------------------------------------

rename *, lower
rename state state_abr
rename countyname county_name
destring statefp, gen(state_fips)
destring countycode, gen(county_fips)

keep state_abr state_fips county_name county_fips year sfr eqi growth
order state_abr state_fips county_name county_fips year sfr eqi growth

*-------------------------------------------------------------------------------
* Fix ZIP-to-County Crosswalk Artifacts
*-------------------------------------------------------------------------------

* Round SFR to integer (artifact from weighted crosswalk)
replace sfr = ceil(sfr)

* Correct masked positive values
replace sfr = 1 if sfr == 0 & eqi > 0

* Convert growth to binary indicator
replace growth = growth > 0

*-------------------------------------------------------------------------------
* Create Balanced Panel
*-------------------------------------------------------------------------------

xtset county_fips year
tsfill, full

* Carry forward/backward state identifiers
hashsort county_fips -year
by county_fips: carryforward state_abr state_fips county_name, replace
hashsort county_fips year
by county_fips: carryforward state_abr state_fips county_name, replace

* Fill missing outcomes with zeros
foreach v of varlist sfr eqi growth {
	replace `v' = 0 if missing(`v')
}

*-------------------------------------------------------------------------------
* Finalize and Save
*-------------------------------------------------------------------------------

keep state_abr state_fips county_name county_fips year sfr eqi growth
order state_abr state_fips county_name county_fips year sfr eqi growth
hashsort county_fips year

* Clear variable labels
foreach v of varlist * {
	label var `v' ""
}

save "${intermediate}/scp.dta", replace


/*******************************************************************************
*                                                                              *
*   SECTION 3: CREATE COUNTY-LEVEL COVARIATES                                  *
*                                                                              *
*   This section creates baseline covariates from:                             *
*     - Historical population data (1900-2010)                                 *
*     - QCEW employment and wage data (1990)                                   *
*                                                                              *
*******************************************************************************/

*===============================================================================
* 3A: HISTORICAL POPULATION DATA (1900-2010)
*===============================================================================

import delimited "${raw}/historical_county_populations_v2.csv", clear varn(1)

rename cty_fips county_fips
keep county_fips pop_1990

* Log-transform population
replace pop_1990 = log(pop_1990)

save "${intermediate}/covariates_pop.dta", replace

*===============================================================================
* 3B: QCEW EMPLOYMENT AND WAGE DATA (1990)
*===============================================================================

* Load US mainland data
import excel "${raw}/allhlcn90.xlsx", clear firstrow sheet("US_St_Cn_MSA")
save "${intermediate}/empwage_temp.dta", replace

* Append Puerto Rico and Virgin Islands
import excel "${raw}/allhlcn90.xlsx", clear firstrow sheet("US_PR_VI")
append using "${intermediate}/empwage_temp.dta"
erase "${intermediate}/empwage_temp.dta"

*-------------------------------------------------------------------------------
* Process Geographic Identifiers
*-------------------------------------------------------------------------------

gen s = St + Cnty
destring St, gen(state_fips) force
destring s, gen(county_fips) force
split Area, parse(",")
rename Area1 county_name
rename Area2 state
drop if missing(county_fips)
keep if AreaType == "County"

*-------------------------------------------------------------------------------
* Create Aggregate Employment/Wage Variables by Ownership Type
*-------------------------------------------------------------------------------

* Total Covered Employment
gen total_est = AnnualAverageEstablishmentCou if Ownership == "Total Covered"
gen total_emp = AnnualAverageEmployment if Ownership == "Total Covered"
gen total_wage = AnnualAveragePay if Ownership == "Total Covered"

* Federal Government
gen fed_est = AnnualAverageEstablishmentCou if Ownership == "Federal Government"
gen fed_emp = AnnualAverageEmployment if Ownership == "Federal Government"
gen fed_wage = AnnualAveragePay if Ownership == "Federal Government"
gen fed_ratio_e = EmploymentLocationQuotientRel if Ownership == "Federal Government"
gen fed_ratio_w = TotalWageLocationQuotientRel if Ownership == "Federal Government"

* State Government
gen state_est = AnnualAverageEstablishmentCou if Ownership == "State Government"
gen state_emp = AnnualAverageEmployment if Ownership == "State Government"
gen state_wage = AnnualAveragePay if Ownership == "State Government"
gen state_ratio_e = EmploymentLocationQuotientRel if Ownership == "State Government"
gen state_ratio_w = TotalWageLocationQuotientRel if Ownership == "State Government"

* Local Government
gen local_est = AnnualAverageEstablishmentCou if Ownership == "Local Government"
gen local_emp = AnnualAverageEmployment if Ownership == "Local Government"
gen local_wage = AnnualAveragePay if Ownership == "Local Government"
gen local_ratio_e = EmploymentLocationQuotientRel if Ownership == "Local Government"
gen local_ratio_w = TotalWageLocationQuotientRel if Ownership == "Local Government"

* Private Sector (All Industries)
gen priv_est = AnnualAverageEstablishmentCou if Ownership == "Private" & Industry == "Total, all industries"
gen priv_emp = AnnualAverageEmployment if Ownership == "Private" & Industry == "Total, all industries"
gen priv_wage = AnnualAveragePay if Ownership == "Private" & Industry == "Total, all industries"
gen priv_ratio_e = EmploymentLocationQuotientRel if Ownership == "Private" & Industry == "Total, all industries"
gen priv_ratio_w = TotalWageLocationQuotientRel if Ownership == "Private" & Industry == "Total, all industries"

*-------------------------------------------------------------------------------
* Create Industry-Specific Variables (Private Sector)
*-------------------------------------------------------------------------------

* Standardize industry names
replace Industry = lower(trim(Industry))
tab Industry if Industry != "Total, all industries"

replace Industry = "education" if regexm(Industry, "education")
replace Industry = "finance" if regexm(Industry, "financial")
replace Industry = "goods" if regexm(Industry, "goods")
replace Industry = "information" if regexm(Industry, "information")
replace Industry = "leisure" if regexm(Industry, "leisure")
replace Industry = "manufacturing" if regexm(Industry, "manufacturing")
replace Industry = "mining" if regexm(Industry, "mining")
replace Industry = "other" if regexm(Industry, "other")
replace Industry = "business" if regexm(Industry, "business")
replace Industry = "service" if regexm(Industry, "service-providing")
replace Industry = "trade" if regexm(Industry, "trade")

* Generate sector-specific variables
glevelsof Industry if Industry != "total, all industries", local(sectors)
foreach s in `sectors' {
    local vname = subinstr("`s'", " ", "_", .)
    gen `vname'_est     = AnnualAverageEstablishmentCou if Ownership == "Private" & Industry == "`s'"
    gen `vname'_emp     = AnnualAverageEmployment       if Ownership == "Private" & Industry == "`s'"
    gen `vname'_wage    = AnnualAveragePay              if Ownership == "Private" & Industry == "`s'"
    gen `vname'_ratio_e = EmploymentLocationQuotientRel if Ownership == "Private" & Industry == "`s'"
    gen `vname'_ratio_w = TotalWageLocationQuotientRel  if Ownership == "Private" & Industry == "`s'"
}

*-------------------------------------------------------------------------------
* Collapse to County Level
*-------------------------------------------------------------------------------

gcollapse (firstnm) total_est-trade_ratio_w, by(state county_name state_fips county_fips)

* Log-transform counts and create missing indicators
foreach v of varlist *_est *_emp *_wage {
	replace `v' = 0 if missing(`v')
	rename `v' temp
	gen `v' = temp
	replace `v' = log(`v') if `v' > 0
	gen has_`v' = `v' > 0
	drop temp
}
drop has_total_*

* Process location quotients
foreach v of varlist *_ratio_* {
	replace `v' = 0 if missing(`v')
	rename `v' temp
	gen `v' = temp
	gen has_`v' = `v' > 0
	drop temp
}

save "${intermediate}/empwage.dta", replace

*===============================================================================
* 3C: MERGE COVARIATE DATASETS
*===============================================================================

* Load employment/wage data
use "${intermediate}/empwage.dta", clear

* Merge with population covariates
merge 1:1 county_fips using "${intermediate}/covariates_pop.dta", keep(3) nogen

* Prefix all covariates with X_
foreach v of varlist total_est-pop_1990 {
	rename `v' X_`v'
}

* Clear variable labels
foreach v of varlist * {
	label var `v' ""
}

save "${intermediate}/covariates.dta", replace

/*
NOTE: The following counties have missing covariates:
    1) Shannon County, South Dakota
    2) Bedford City, Virginia
    3) Prince of Wales-Outer Ketchikan Census Area, Alaska
    4) Valdez-Cordova Census Area, Alaska
    5) Wade Hampton Census Area, Alaska
    6) Wrangell-Petersburg Census Area, Alaska
    7) Puerto Rico
    8) U.S. Virgin Islands
*/


/*******************************************************************************
*                                                                              *
*   SECTION 4: MERGE ALL COUNTY-LEVEL DATA                                     *
*                                                                              *
*   This section combines:                                                     *
*     - Startup Cartography Project outcomes                                   *
*     - State tax incentive treatments                                         *
*     - County-level covariates                                                *
*                                                                              *
*******************************************************************************/

*-------------------------------------------------------------------------------
* Load SCP County Data
*-------------------------------------------------------------------------------

use "${intermediate}/scp.dta", clear

*-------------------------------------------------------------------------------
* Merge State-Level Tax Incentives
*-------------------------------------------------------------------------------

merge m:1 state_abr year using "${intermediate}/pdit.dta", keep(1 3)

* Keep only counties that never merged (i.e., non-US states)
hashsort county_fips year
by county_fips: gegen max_merge = max(_merge == 3)
keep if max_merge == 1
drop _merge max_merge

* Forward-fill treatment variables
hashsort county_fips -year
by county_fips: carryforward rnd itc G_rnd G_itc G_both ///
	treat_rnd treat_itc treat_both ///
	treat_post_rnd treat_post_itc treat_post_both, replace
hashsort county_fips year
by county_fips: carryforward rnd itc G_rnd G_itc G_both ///
	treat_rnd treat_itc treat_both ///
	treat_post_rnd treat_post_itc treat_post_both, replace
	
*-------------------------------------------------------------------------------
* Merge County-Level Covariates
*-------------------------------------------------------------------------------

* Handle Broomfield County (created from Boulder County in 2001)
gen Broomfield = (county_fips == 8014)
replace county_fips = 8013 if county_fips == 8014

merge m:1 county_fips using "${intermediate}/covariates.dta", keep(3) nogen

* Restore Broomfield FIPS code
replace county_fips = 8014 if Broomfield == 1
drop Broomfield

*-------------------------------------------------------------------------------
* Create Baseline Outcome Covariates
*-------------------------------------------------------------------------------

* Get baseline year values for covariates
hashsort county_fips year
by county_fips: gen first_year = year[1]
su first_year
local baseline = r(min)

* Create baseline outcome measures
gen temp_sfr = sfr if year == `baseline'
gen temp_eqi = eqi if year == `baseline'
gen temp_growth = growth if year == `baseline'

by county_fips: gegen X_sfr_base = firstnm(temp_sfr)
by county_fips: gegen X_eqi_base = firstnm(temp_eqi)
by county_fips: gegen X_growth_base = firstnm(temp_growth)
drop temp_* first_year

* Transform baseline measures
gen X_sfr = log(X_sfr_base) if X_sfr_base > 0
replace X_sfr = 0 if missing(X_sfr)

gen X_eqi = log(X_eqi_base / (1 - X_eqi_base)) if X_eqi_base > 0 & X_eqi_base < 1
replace X_eqi = 0 if missing(X_eqi)

gen X_growth = X_growth_base
gen X_has_sfr = X_sfr_base > 0

drop X_sfr_base X_eqi_base X_growth_base

*-------------------------------------------------------------------------------
* Create 10-Year Average Treatment Intensity
*-------------------------------------------------------------------------------

* R&D Credit: average over first 15 years post-adoption
hashsort county_fips year
gen t = rnd if year >= G_rnd & year <= G_rnd + 14
by county_fips: gegen avg15_rnd = mean(t)
drop t
replace avg15_rnd = 0 if missing(avg15_rnd)

* Investment Tax Credit: average over first 15 years post-adoption
hashsort county_fips year
gen t = itc if year >= G_itc & year <= G_itc + 14
by county_fips: gegen avg15_itc = mean(t)
drop t
replace avg15_itc = 0 if missing(avg15_itc)

*-------------------------------------------------------------------------------
* Create Transformed Outcome Variables
*-------------------------------------------------------------------------------

* Log transformation (zero = 0)
gen log_sfr = 0
replace log_sfr = log(sfr) if sfr > 0

* Log(1 + Y) transformation
gen log1p_sfr = log(1 + sfr)

* Per 1,000 population transformation
gen sfr_per1k = 1000 * (sfr / ceil(exp(X_pop_1990)))

*-------------------------------------------------------------------------------
* Organize Final Dataset
*-------------------------------------------------------------------------------

hashsort state_fips county_fips year

keep state_abr state_fips county_fips year ///
	G_rnd rnd treat_rnd treat_post_rnd avg15_rnd ///
	G_itc itc treat_itc treat_post_itc avg15_itc ///
	G_both treat_both treat_post_both ///
	sfr log_sfr log1p_sfr sfr_per1k eqi growth ///
	X_*

order state_abr state_fips county_fips year ///
	G_rnd rnd treat_rnd treat_post_rnd avg15_rnd ///
	G_itc itc treat_itc treat_post_itc avg15_itc ///
	G_both treat_both treat_post_both ///
	sfr log_sfr log1p_sfr sfr_per1k eqi growth ///
	X_*

* Clear variable labels
foreach v of varlist * {
	label var `v' ""
}

*-------------------------------------------------------------------------------
* Create Sample Indicators (Exclude Always-Treated and Small Cohorts)
*-------------------------------------------------------------------------------

gen Sample_rnd  = !inlist(G_rnd, 1990, 1992, 1993)
gen Sample_itc  = !inlist(G_itc, 1990)
gen Sample_both = !inlist(G_both, 1990, 1993)

order Sample_*, first

*-------------------------------------------------------------------------------
* Create Parsimonious Covariate Set (Z variables)
*-------------------------------------------------------------------------------

* For comparison with full covariate specifications
gen Z1 = X_pop_1990          // Log population (1990)
gen Z2 = X_sfr               // Log baseline SFR
gen Z3 = X_has_sfr           // Any startups indicator
gen Z4 = X_eqi               // Baseline EQI (logit)
gen Z5 = X_growth            // Baseline growth indicator

order Z1 Z2 Z3 Z4 Z5, last

/*******************************************************************************
*                                                                              *
*   SECTION 5: EXPORT FINAL DATASET                                            *
*                                                                              *
*******************************************************************************/

save "${final}/scp_pdit_county.dta", replace
drop X_has_*
export delimited "${final}/scp_pdit_county.csv", replace

* End of script


***** Create Cohort-Specific Datasets

use "${final}/scp_pdit_county.dta", clear
keep if Sample_rnd == 1
keep county_fips G_rnd
gduplicates drop
export delimited "${final}/csv/ids.csv", replace

use "${final}/scp_pdit_county.dta", clear
keep if Sample_rnd == 1
replace G_rnd = 0 if G_rnd > 2010
keep if inrange(year, 1988, 2010)
glevelsof G_rnd if G_rnd != 0, local(cohort_list)
foreach g in `cohort_list' {
	dis "`g'"
	preserve
		
	keep if G_rnd == 0 | G_rnd == `g'
	keep if inrange(year, `g' - 1 - 5, `g' - 1 + 10)
	summarize year
	
	local min_year = `r(min)' + 1
	local max_year = `r(max)'
	
	local n_leads = `g' - `min_year'
	local n_lags  = `max_year' - `g' + 1
	
	local base_year = `g' - 1
	
	local f = `n_leads'
	forvalues i = `min_year'(1)`base_year' {
		local j = `i' - 1
		gen temp_pre  = sfr if year == `i'
		gen temp_post = sfr if year == `j'
		hashsort county_fips year
		by county_fips: gegen lead_pre_`f'  = firstnm(temp_pre)
		by county_fips: gegen lead_post_`f' = firstnm(temp_post)
		drop temp_pre temp_post
		local f = `f' - 1
	}
	
	gen temp_base = sfr if year == `base_year'
	hashsort county_fips year
	by county_fips: gegen lag_pre_base = firstnm(temp_base)
	drop temp_base

	local k = 0
	forvalues i = `g'(1)`max_year' {
		gen temp_post = sfr if year == `i'
		hashsort county_fips year
		by county_fips: gegen lag_post_`k' = firstnm(temp_post)
		drop temp_post
		local k = `k' + 1
	}
	
	gen D_treat = (G_rnd == `g')
	drop X_has_*
	gcollapse (firstnm) lead_* lag_* X* Z*, by(state_fips county_fips D_treat)
	foreach v of varlist lead_* lag_* {
		rename `v' Y_`v'
	}
	
	gstats winsor Y_*, by(D_treat) cut(0 90) replace
	export delimited "${final}/csv/cohort_`g'.csv", replace
	
	restore
}

