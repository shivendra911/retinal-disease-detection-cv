"""
IRDAS Stage 3 — Multilingual Patient Report Generation
========================================================

Converts model outputs into plain-language patient reports
in Indian languages using Gemini API via LangChain.

Supported languages: Hindi, Tamil, Telugu, Bengali, Marathi,
                     Kannada, Malayalam, Punjabi, English

The report is NOT a clinical document — it's a human-friendly explanation
that a non-medical, potentially illiterate patient can understand.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


DR_SEVERITY = {
    0: 'no diabetic eye disease',
    1: 'mild diabetic eye changes',
    2: 'moderate diabetic eye damage',
    3: 'severe diabetic eye damage',
    4: 'advanced diabetic eye disease requiring urgent treatment'
}

HR_SEVERITY = {
    0: 'no blood pressure related eye changes',
    1: 'mild blood pressure related changes',
    2: 'moderate blood pressure related eye damage'
}

LANGUAGE_NAMES = {
    'hindi': 'Hindi', 'tamil': 'Tamil', 'telugu': 'Telugu',
    'bengali': 'Bengali', 'marathi': 'Marathi', 'kannada': 'Kannada',
    'malayalam': 'Malayalam', 'punjabi': 'Punjabi', 'english': 'English'
}


def generate_patient_report(dr_grade, hr_present, dr_uncertainty,
                            hr_uncertainty, heatmap_description,
                            patient_language='hindi', gemini_api_key=None):
    """
    Generate a patient-facing explanation of retinal scan results.
    
    Args:
        dr_grade: int 0-4, DR severity grade
        hr_present: float 0-1, HR probability
        dr_uncertainty: float, MC Dropout uncertainty for DR
        hr_uncertainty: float, MC Dropout uncertainty for HR
        heatmap_description: str from describe_heatmap_regions()
        patient_language: str, target language key
        gemini_api_key: str, Google AI Studio API key
    
    Returns:
        str: Patient report in the specified language
    """
    llm = ChatGoogleGenerativeAI(
        model='gemini-pro',
        google_api_key=gemini_api_key,
        temperature=0.3  # low temperature for consistent medical communication
    )
    
    confidence_str = (
        'high confidence' if dr_uncertainty < 0.1
        else 'moderate confidence — a specialist review is recommended'
    )
    hr_str = HR_SEVERITY.get(int(hr_present > 0.5), HR_SEVERITY[0])
    dr_str = DR_SEVERITY.get(int(dr_grade), DR_SEVERITY[0])
    lang_name = LANGUAGE_NAMES.get(patient_language, 'English')
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a compassionate health worker explaining a retinal scan
         result to a patient in {language}. Use very simple words. No medical jargon.
         The patient may be illiterate or semi-literate. Be kind and clear.
         Write ONLY in {language}. Maximum 120 words total."""),
        ("human", """Eye scan findings:
         - Diabetes-related: {dr_finding}
         - Blood pressure-related: {hr_finding}
         - Where in the eye: {location}
         - Reliability: {confidence}

         Please tell the patient:
         1. What was found (in one simple sentence)
         2. What this means for their daily life
         3. Exactly what they must do next (be specific: 'go to eye hospital within X days')""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        'language'   : lang_name,
        'dr_finding' : dr_str,
        'hr_finding' : hr_str,
        'location'   : heatmap_description,
        'confidence' : confidence_str,
    })
    return response.content


def generate_offline_report(dr_grade, hr_present, dr_uncertainty, patient_language='english'):
    """
    Offline fallback when Gemini API is unavailable.
    Returns a template-based report in English.
    """
    severity = DR_SEVERITY.get(dr_grade, DR_SEVERITY[0])
    hr_status = "present" if hr_present > 0.5 else "not detected"
    confidence = "high" if dr_uncertainty < 0.1 else "moderate"
    
    urgency = {
        0: "No action needed. Recheck in 12 months.",
        1: "Schedule an eye checkup within 6 months.",
        2: "See an eye specialist within 1 month.",
        3: "See an eye specialist within 1 week. This is urgent.",
        4: "Go to an eye hospital TODAY. This needs immediate treatment."
    }
    
    return (
        f"Eye Scan Results\n"
        f"================\n"
        f"Diabetes eye check: {severity}\n"
        f"Blood pressure eye changes: {hr_status}\n"
        f"Confidence level: {confidence}\n\n"
        f"What to do: {urgency.get(dr_grade, urgency[0])}"
    )
