def validate_patient(patient):
    
    errors = []
    
    if patient["oxygen_support"] and patient["ECOG"] == 0:
        errors.append("oxygen_support плохо сочетается с ECOG 0")

    if patient["oxygen_support"] and "одышка" not in patient.get("symptoms", []):
        errors.append("oxygen_support лучше сочетать с симптомом 'одышка'")

    if patient["weight_loss"] and patient["BMI"] > 30:
        errors.append("weight_loss плохо сочетается с высоким BMI")

    if patient["BMI"] < 18.5 and not patient["weight_loss"]:
        errors.append("низкий BMI лучше сочетать с weight_loss=True")

    if patient["hemoglobin"] < 100 and patient["ECOG"] == 0:
        errors.append("выраженная анемия плохо сочетается с ECOG 0")

    if patient["creatinine"] > 120 and patient["renal_function"] == "normal":
        errors.append("высокий креатинин плохо сочетается с normal renal_function")

    if patient["AST"] > 60 and patient["liver_function"] == "normal":
        errors.append("повышенный AST плохо сочетается с normal liver_function")

    if patient["ALT"] > 60 and patient["liver_function"] == "normal":
        errors.append("повышенный ALT плохо сочетается с normal liver_function")

    if patient["drug_contraindications"] == "platinum" and patient.get("cisplatin_eligible") is True:
        errors.append("platinum contraindication не должен сочетаться с cisplatin_eligible=True")

    if patient["drug_contraindications"] == "immunotherapy" and patient.get("immunotherapy_eligible") is True:
        errors.append("immunotherapy contraindication не должен сочетаться с immunotherapy_eligible=True")

    if patient["pleural_effusion"] and "одышка" not in patient.get("symptoms", []):
        errors.append("плевральный выпот лучше сочетать с одышкой")

    if patient["hemoptysis"] and "кашель" not in patient.get("symptoms", []):
        errors.append("кровохарканье лучше сочетать с кашлем")

    return errors

def validate_case(case):
    
    errors = []

    p = case["patient"]
    d = case["diagnosis"]
    c = case["clinical_context"]

    errors.extend(validate_patient(p))

    # stage / TNM / metastases
    if d["stage"] != "IV" and d["metastases"]:
        errors.append("Метастазы допустимы только при IV стадии")

    if d["stage"] == "IV" and not d["metastases"]:
        errors.append("При IV стадии должны быть указаны метастазы")

    if d["stage"] != "IV" and "M1" in d["TNM"]:
        errors.append("M1 в TNM допустим только при IV стадии")

    if d["stage"] == "IV" and "M0" in d["TNM"]:
        errors.append("M0 не соответствует IV стадии")

    # resectability
    if d["stage"] == "IV" and d["resectability"] != "unresectable":
        errors.append("IV стадия обычно должна быть unresectable")

    if d["stage"] in ["I", "II"] and d["resectability"] == "unresectable":
        errors.append("I-II стадия редко должна быть unresectable")

    if d["stage"] == "III" and d["resectability"] == "resectable" and "N3" in d["TNM"]:
        errors.append("N3 плохо сочетается с resectable")

    # metastatic burden
    if d["stage"] != "IV" and d["metastatic_burden"] != "not_applicable":
        errors.append("metastatic_burden должен быть not_applicable при I-III стадии")

    if d["stage"] == "IV" and d["metastatic_burden"] == "not_applicable":
        errors.append("metastatic_burden должен быть указан при IV стадии")

    if d["metastatic_burden"] == "oligometastatic" and len(d["metastases"]) > 2:
        errors.append("oligometastatic плохо сочетается с >2 локализациями метастазов")

    if d["metastatic_burden"] == "multiple" and len(d["metastases"]) == 1:
        errors.append("multiple metastatic burden плохо сочетается с одной локализацией")

    # tumor size vs TNM грубо
    if d["tumor_size_cm"] <= 3 and d["TNM"].startswith("T4"):
        errors.append("малый размер опухоли плохо сочетается с T4")

    if d["tumor_size_cm"] > 7 and d["TNM"].startswith("T1"):
        errors.append("размер >7 см плохо сочетается с T1")

    # cancer type / markers
    if d["type"] == "мелкоклеточный рак легкого":
        for marker in ["EGFR", "ALK", "ROS1", "BRAF", "PD-L1"]:
            if d[marker] != "not_applicable":
                errors.append(f"{marker} для МРЛ должен быть not_applicable")

    if d["type"] == "немелкоклеточный рак легкого":
        positive_drivers = [m for m in ["EGFR", "ALK", "ROS1", "BRAF"] if d[m] == "positive"]
        if len(positive_drivers) > 1:
            errors.append(f"Драйверные мутации должны быть взаимоисключающими: {positive_drivers}")

    # context
    previous = c["previous_treatment"]
    response = c["treatment_response"]
    progression = c["progression_pattern"]
    line = c["treatment_line"]
    goal = c["clinical_goal"]

    if previous == "не получал" and response != "not_applicable":
        errors.append("Если лечения не было, treatment_response должен быть not_applicable")

    if previous == "не получал" and progression != "not_applicable":
        errors.append("Если лечения не было, progression_pattern должен быть not_applicable")

    if previous == "не получал" and line != "first_line":
        errors.append("Если лечения не было, treatment_line должен быть first_line")

    if response == "progression" and progression == "not_applicable":
        errors.append("При progression должен быть указан progression_pattern")

    if "прогрессирование" in previous and goal != "treatment_after_progression":
        errors.append("При прогрессировании clinical_goal лучше treatment_after_progression")

    if previous == "прогрессирование после таргетной терапии":
        has_driver = any(d[m] == "positive" for m in ["EGFR", "ALK", "ROS1", "BRAF"])
        if not has_driver:
            errors.append("Прогрессирование после таргетной терапии требует драйверной мутации")

    if c["progression_pattern"] == "CNS_only" and "головной мозг" not in d["metastases"]:
        errors.append("CNS_only progression требует метастазов в головной мозг")

    return errors