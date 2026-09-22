-- ====================================================================
-- MIMIC-IV 24-Hour Sequential Vital Signs & Laboratory Extraction
-- Extracts hourly aggregate trajectories for Heart Rate, RR, MAP, Temp, WBC
-- ====================================================================

WITH icu_grid AS (
    SELECT 
        stay_id,
        intime,
        TIMESTAMP_ADD(intime, INTERVAL hr HOUR) AS hr_start,
        TIMESTAMP_ADD(intime, INTERVAL hr + 1 HOUR) AS hr_end,
        hr AS hour_step
    FROM `physionet-data.mimiciv_icu.icustays`,
    UNNEST(GENERATE_ARRAY(0, 23)) AS hr
),

vitals AS (
    SELECT 
        g.stay_id,
        g.hour_step,
        AVG(CASE WHEN ce.itemid IN (220045) THEN ce.valuenum END) AS heart_rate,
        AVG(CASE WHEN ce.itemid IN (220210, 224690) THEN ce.valuenum END) AS resp_rate,
        AVG(CASE WHEN ce.itemid IN (220052, 220181, 225312) THEN ce.valuenum END) AS mean_arterial_pressure,
        AVG(CASE WHEN ce.itemid IN (223761, 223762) THEN (CASE WHEN ce.itemid = 223761 THEN (ce.valuenum - 32) * 5/9 ELSE ce.valuenum END) END) AS temperature
    FROM icu_grid g
    LEFT JOIN `physionet-data.mimiciv_icu.chartevents` ce
        ON g.stay_id = ce.stay_id
        AND ce.charttime >= g.hr_start 
        AND ce.charttime < g.hr_end
    WHERE ce.itemid IN (220045, 220210, 224690, 220052, 220181, 225312, 223761, 223762)
    GROUP BY g.stay_id, g.hour_step
),

labs AS (
    SELECT 
        g.stay_id,
        g.hour_step,
        AVG(CASE WHEN le.itemid IN (51301, 51300) THEN le.valuenum END) AS wbc_count
    FROM icu_grid g
    LEFT JOIN `physionet-data.mimiciv_hosp.labevents` le
        ON g.stay_id = le.stay_id
        AND le.charttime >= g.hr_start 
        AND le.charttime < g.hr_end
    WHERE le.itemid IN (51301, 51300)
    GROUP BY g.stay_id, g.hour_step
)

SELECT 
    g.stay_id,
    g.hour_step,
    COALESCE(v.heart_rate, 80.0) AS heart_rate,
    COALESCE(v.resp_rate, 16.0) AS resp_rate,
    COALESCE(v.mean_arterial_pressure, 85.0) AS mean_arterial_pressure,
    COALESCE(v.temperature, 37.0) AS temperature,
    COALESCE(l.wbc_count, 8.0) AS wbc_count
FROM icu_grid g
LEFT JOIN vitals v ON g.stay_id = v.stay_id AND g.hour_step = v.hour_step
LEFT JOIN labs l ON g.stay_id = l.stay_id AND g.hour_step = l.hour_step
ORDER BY g.stay_id, g.hour_step;
