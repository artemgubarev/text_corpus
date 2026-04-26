import json
import clinical_context as cl
import patient_data as pt
import diagnosis_data as dg
from validator import validate_case

def generate_case():
    patient = pt.generate_patient()
    diagnosis = dg.generate_diagnosis()
    clinical_context = cl.generate_clinical_context(diagnosis["stage"])

    return {
        "patient": patient,
        "diagnosis": diagnosis,
        "clinical_context": clinical_context
    }
    
def case_to_text(case):
    patient_text = pt.text_describe_patient(case["patient"])
    diagnosis_text = dg.text_describe_diagnosis(case["diagnosis"])
    context_text = cl.text_describe_clinical_context(case["clinical_context"])

    return " ".join([
        patient_text,
        diagnosis_text,
        context_text
    ])

texts = []

for i in range(0, 100):
    case = generate_case()
    errors = validate_case(case)
    if not errors:
        text = case_to_text(case)
        texts.append(text)

with open("../../data/synthetic_data.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)