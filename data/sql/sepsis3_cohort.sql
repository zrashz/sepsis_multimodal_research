-- ====================================================================
-- MIMIC-IV Sepsis-3 Cohort Selection Query
-- Inclusion Criteria: Adult ICU patients (Age >= 18) with Sepsis-3 definition
-- ====================================================================

WITH icu_adults AS (
    SELECT 
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime,
        ie.outtime,
        pat.gender,
        pat.anchor_age AS age
    FROM `physionet-data.mimiciv_icu.icustays` ie
    INNER JOIN `physionet-data.mimiciv_hosp.patients` pat
        ON ie.subject_id = pat.subject_id
    WHERE pat.anchor_age >= 18
),

sepsis3_patients AS (
    SELECT 
        stay_id,
        sofa_score,
        suspected_infection_time,
        sepsis3 AS is_sepsis
    FROM `physionet-data.mimiciv_derived.sepsis3`
)

SELECT 
    a.subject_id,
    a.hadm_id,
    a.stay_id,
    a.age,
    CASE WHEN a.gender = 'M' THEN 1 ELSE 0 END AS gender,
    COALESCE(s.is_sepsis, 0) AS sepsis_label,
    a.intime AS icu_intime
FROM icu_adults a
LEFT JOIN sepsis3_patients s
    ON a.stay_id = s.stay_id;
