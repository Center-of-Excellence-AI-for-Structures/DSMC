import numpy as np


def calculate_sofa_score(
    respiratory_rate,
    platelet_count,
    mean_arterial_pressure,
    bilirubin,
    Glasgow_coma_scale,
    creatinine,
):
    """
    after calculating the SOFA score, the probability of in-hospital mortality can be calculated in a discrete way according to the literature
    """

    sofa_score = 0

    # Respiratory rate
    if respiratory_rate < 400:
        sofa_score += 1
    elif respiratory_rate < 300:
        sofa_score += 2
    elif respiratory_rate < 200:
        sofa_score += 3
    elif respiratory_rate < 100:
        sofa_score += 4

    # Platelet Count
    if platelet_count < 150:
        sofa_score += 1
    elif platelet_count < 100:
        sofa_score += 2
    elif platelet_count < 50:
        sofa_score += 3
    elif platelet_count < 20:
        sofa_score += 4

    # Mean Arterial Pressure
    if mean_arterial_pressure < 70:
        sofa_score += 1

    # Bilirubin
    if 1.2 <= bilirubin <= 1.9:
        sofa_score += 1
    elif 2.0 <= bilirubin <= 5.9:
        sofa_score += 2
    elif 6.0 <= bilirubin <= 11.9:
        sofa_score += 3
    elif bilirubin > 12.0:
        sofa_score += 4

    # Glasgow Coma Scale
    if 13 <= Glasgow_coma_scale <= 14:
        sofa_score += 1
    elif 10 <= Glasgow_coma_scale <= 12:
        sofa_score += 2
    elif 6 <= Glasgow_coma_scale <= 9:
        sofa_score += 3
    elif Glasgow_coma_scale < 6:
        sofa_score += 4

    # Creatinine
    if 1.2 <= creatinine <= 1.9:
        sofa_score += 1
    elif 2.0 <= creatinine <= 3.4:
        sofa_score += 2
    elif 3.5 <= creatinine <= 4.9:
        sofa_score += 3
    elif creatinine > 5.0:
        sofa_score += 4

    if sofa_score <= 1:
        prob = 0.0
    elif 2 <= sofa_score <= 3:
        prob = 6.4
    elif 4 <= sofa_score <= 5:
        prob = 20.2
    elif 6 <= sofa_score <= 7:
        prob = 21.5
    elif 8 <= sofa_score <= 9:
        prob = 33.3
    elif 10 <= sofa_score <= 11:
        prob = 50.0
    else:
        prob = 95.2

    return prob, sofa_score


def calculate_saps_iii_score(
    age,
    heart_rate,
    bilirubin,
    mean_arterial_pressure,
    body_temperature,
    Glasgow_coma_scale,
    platelets,
    creatinine,
    respiratory_rate,
):
    """
    after calculating the SAPS III score, the probability of in-hospital mortality can be calculated as:
    % = e^x / 1 + e^x, where:
    x =  −32.6659 + ln(saps_iii_score + 20.5958) × 7.3068

    literature: https://pubmed.ncbi.nlm.nih.gov/16132892/
    """

    saps_iii_score = 0

    # Age
    if age >= 80:
        saps_iii_score += 18
    elif age >= 75:
        saps_iii_score += 15
    elif age >= 70:
        saps_iii_score += 13
    elif age >= 60:
        saps_iii_score += 9
    elif age >= 40:
        saps_iii_score += 5

    # Heart Rate
    if heart_rate < 120:
        saps_iii_score += 1
    elif heart_rate <= 159:
        saps_iii_score += 5
    else:
        saps_iii_score += 7

    # Bilirubin
    if bilirubin < 2:
        saps_iii_score += 1
    elif bilirubin <= 5.9:
        saps_iii_score += 4
    else:
        saps_iii_score += 1

    # Mean Arterial Pressure
    if mean_arterial_pressure < 40:
        saps_iii_score += 12
    elif mean_arterial_pressure <= 69:
        saps_iii_score += 8
    elif mean_arterial_pressure <= 119:
        saps_iii_score += 3
    else:
        saps_iii_score += 1

    # Body Temperature
    if body_temperature < 35:
        saps_iii_score += 7.5
    else:
        saps_iii_score += 1

    # Glasgow Coma Scale
    if Glasgow_coma_scale < 5:
        saps_iii_score += 15
    elif Glasgow_coma_scale == 5:
        saps_iii_score += 10
    elif Glasgow_coma_scale == 6:
        saps_iii_score += 7.5
    elif Glasgow_coma_scale <= 12:
        saps_iii_score += 2
    else:
        saps_iii_score += 1

    # Platelets
    if platelets < 20:
        saps_iii_score += 13
    elif platelets <= 49:
        saps_iii_score += 8
    elif platelets <= 99:
        saps_iii_score += 5
    else:
        saps_iii_score += 1

    # Serum Creatinine
    if creatinine < 1.2:
        saps_iii_score += 1
    elif creatinine <= 1.19:
        saps_iii_score += 2
    elif creatinine <= 3.4:
        saps_iii_score += 7
    else:
        saps_iii_score += 8

    # Respiratory rate (PaO2/FiO2)
    if 60 < respiratory_rate < 100:
        saps_iii_score += 11
    elif respiratory_rate >= 100:
        saps_iii_score += 7
    else:
        saps_iii_score += 1

    x = -32.6659 + np.log(saps_iii_score + 20.5958) * 7.3068
    prob = np.exp(x) / (1 + np.exp(x))

    return prob, saps_iii_score


def calculate_apache_ii_score(
    age,
    heart_rate,
    mean_arterial_pressure,
    body_temperature,
    resipiratory_rate,
    Glasgow_coma_scale,
    potassium,
    sodium,
    creatinine,
    hematocrit,
):
    """
    after calculating the APACHE II score, the probability of in-hospital mortality can be calculated as:
    % = e^x / 1 + e^x, where:
    x =  −	3.517 + APACHE II × 0.146

    literature: https://europepmc.org/article/med/3928249?utm_campaign=share&utm_medium=email&utm_source=email_share_mailer&client=bot&client=bot
    """

    apache_ii_score = 0

    # Age
    if age >= 75:
        apache_ii_score += 6
    elif age >= 65:
        apache_ii_score += 5
    elif age >= 55:
        apache_ii_score += 3
    else:
        apache_ii_score += 2

    # Heart Rate
    if heart_rate >= 180 or heart_rate <= 39:
        apache_ii_score += 4
    elif 140 <= heart_rate <= 179 or 40 <= heart_rate <= 54:
        apache_ii_score += 3
    elif 110 <= heart_rate <= 139 or 55 <= heart_rate <= 69:
        apache_ii_score += 2

    # Mean Arterial Pressure
    if mean_arterial_pressure <= 49 or mean_arterial_pressure >= 160:
        apache_ii_score += 4
    elif 50 <= mean_arterial_pressure <= 69 or 110 <= mean_arterial_pressure <= 129:
        apache_ii_score += 2
    elif 130 <= mean_arterial_pressure <= 159:
        apache_ii_score += 3

    # Body Temperature
    if body_temperature <= 41 or body_temperature >= 29.9:
        apache_ii_score += 4
    elif 39 <= body_temperature <= 40.9 or 30 <= body_temperature <= 31.9:
        apache_ii_score += 3
    elif 32 <= body_temperature <= 33.9:
        apache_ii_score += 2
    elif 38.5 <= body_temperature <= 38.9 or 34 <= body_temperature <= 35.9:
        apache_ii_score += 1

    # Respiratory rate
    if resipiratory_rate <= 5 or resipiratory_rate >= 41:
        apache_ii_score += 4
    elif 6 <= resipiratory_rate <= 9:
        apache_ii_score += 2
    elif 10 <= resipiratory_rate <= 11 or 25 <= resipiratory_rate <= 34:
        apache_ii_score += 1

    # Glasgow Coma Scale
    apache_ii_score += 15 - Glasgow_coma_scale

    # Potassium
    if potassium < 2.5 or potassium >= 7.0:
        apache_ii_score += 4
    elif 6.0 <= potassium <= 6.9:
        apache_ii_score += 3
    elif 2.5 <= potassium <= 2.9:
        apache_ii_score += 2
    elif 3.0 <= potassium <= 3.4 or 5.5 <= potassium <= 5.9:
        apache_ii_score += 1

    # Sodium
    if sodium <= 110 or sodium >= 180:
        apache_ii_score += 4
    elif 111 <= sodium <= 119 or 160 <= sodium <= 179:
        apache_ii_score += 3
    elif 120 <= sodium <= 129 or 150 <= sodium <= 159:
        apache_ii_score += 2
    elif 150 <= sodium <= 154:
        apache_ii_score += 1

    # Creatinine
    if creatinine >= 3.5:
        apache_ii_score += 4
    elif 2.0 <= creatinine <= 3.4:
        apache_ii_score += 3
    elif 1.5 <= creatinine <= 1.9 or creatinine < 0.6:
        apache_ii_score += 2

    # Hematocrit
    if hematocrit < 20 or hematocrit >= 60:
        apache_ii_score += 4
    elif 20 <= hematocrit <= 29.9 or 50 <= hematocrit <= 59.9:
        apache_ii_score += 2
    elif 46 <= hematocrit <= 49.9:
        apache_ii_score += 1

    x = -3.517 + apache_ii_score * 0.146
    prob = np.exp(x) / (1 + np.exp(x))

    return prob, apache_ii_score
