-- ====================================================================
-- MIMIC-IV Clinical Notes Extraction
-- Pulls nursing notes and progress reports prior to prediction horizon
-- ====================================================================

SELECT 
    ie.stay_id,
    ie.subject_id,
    STRING_AGG(ne.text, ' ') AS text_note
FROM `physionet-data.mimiciv_icu.icustays` ie
INNER JOIN `physionet-data.mimiciv_note.discharge` ne
    ON ie.subject_id = ne.subject_id
    AND ne.charttime >= ie.intime
    AND ne.charttime <= TIMESTAMP_ADD(ie.intime, INTERVAL 24 HOUR)
GROUP BY ie.stay_id, ie.subject_id;
