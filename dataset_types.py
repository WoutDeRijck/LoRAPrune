from pydantic import BaseModel, Field
from enum import Enum

class Diagnosis(str, Enum):
    FLU = "FLU"
    COLD = "COLD"
    COVID19 = "COVID19"
    ASTHMA = "ASTHMA"
    DIABETES = "DIABETES"
    HYPERTENSION = "HYPERTENSION"
    MIGRAINE = "MIGRAINE"
    ANEMIA = "ANEMIA"
    ARTHRITIS = "ARTHRITIS"
    DEPRESSION = "DEPRESSION"
    ALLERGY = "ALLERGY"
    BRONCHITIS = "BRONCHITIS"

    @classmethod
    def description(cls, key):
        descriptions = {
            cls.FLU: "Influenza, commonly known as the flu, is an infectious disease caused by an influenza virus.",
            cls.COLD: "The common cold is a viral infection of your nose and throat (upper respiratory tract).",
            cls.COVID19: "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus.",
            cls.ASTHMA: "Asthma is a condition in which your airways narrow and swell and may produce extra mucus.",
            cls.DIABETES: "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high.",
            cls.HYPERTENSION: "Hypertension, also known as high blood pressure, is a condition in which the blood pressure in the arteries is persistently elevated.",
            cls.MIGRAINE: "Migraine is a neurological condition that can cause multiple symptoms, most notably a severe headache.",
            cls.ANEMIA: "Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues.",
            cls.ARTHRITIS: "Arthritis is the swelling and tenderness of one or more of your joints.",
            cls.DEPRESSION: "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest.",
            cls.ALLERGY: "An allergy is a reaction by your immune system to something that does not bother most other people.",
            cls.BRONCHITIS: "Bronchitis is an inflammation of the lining of your bronchial tubes, which carry air to and from your lungs."
        }
        return descriptions.get(key, "")
class Medication(str, Enum):
    PARACETAMOL = "PARACETAMOL"
    IBUPROFEN = "IBUPROFEN"
    AMOXICILLIN = "AMOXICILLIN"
    METFORMIN = "METFORMIN"
    ALBUTEROL = "ALBUTEROL"
    LISINOPRIL = "LISINOPRIL"
    ATORVASTATIN = "ATORVASTATIN"
    ASPIRIN = "ASPIRIN"
    OMEPRAZOLE = "OMEPRAZOLE"
    LEVOTHYROXINE = "LEVOTHYROXINE"
    SIMVASTATIN = "SIMVASTATIN"
    LOSARTAN = "LOSARTAN"

    @classmethod
    def description(cls, key):
        descriptions = {
            cls.PARACETAMOL: "Paracetamol is a common painkiller used to treat aches and pain.",
            cls.IBUPROFEN: "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for treating pain, fever, and inflammation.",
            cls.AMOXICILLIN: "Amoxicillin is an antibiotic used to treat a number of bacterial infections.",
            cls.METFORMIN: "Metformin is a medication used to treat type 2 diabetes.",
            cls.ALBUTEROL: "Albuterol is a medication that opens up the medium and large airways in the lungs.",
            cls.LISINOPRIL: "Lisinopril is a medication used to treat high blood pressure and heart failure.",
            cls.ATORVASTATIN: "Atorvastatin is a medication used to lower cholesterol and reduce the risk of heart disease.",
            cls.ASPIRIN: "Aspirin is used to reduce pain, fever, or inflammation.",
            cls.OMEPRAZOLE: "Omeprazole is a medication used to treat gastroesophageal reflux disease and other stomach acid-related conditions.",
            cls.LEVOTHYROXINE: "Levothyroxine is a medication used to treat hypothyroidism.",
            cls.SIMVASTATIN: "Simvastatin is a medication used to control high cholesterol.",
            cls.LOSARTAN: "Losartan is a medication used to treat high blood pressure."
        }
        return descriptions.get(key, "")
class Symptom(str, Enum):
    FEVER = "FEVER"
    COUGH = "COUGH"
    FATIGUE = "FATIGUE"
    SHORTNESS_OF_BREATH = "SHORTNESS_OF_BREATH"
    NAUSEA = "NAUSEA"
    HEADACHE = "HEADACHE"
    DIZZINESS = "DIZZINESS"
    SORE_THROAT = "SORE_THROAT"
    RUNNY_NOSE = "RUNNY_NOSE"
    MUSCLE_ACHES = "MUSCLE_ACHES"
    LOSS_OF_TASTE = "LOSS_OF_TASTE"
    LOSS_OF_SMELL = "LOSS_OF_SMELL"
    CHEST_PAIN = "CHEST_PAIN"
    CHILLS = "CHILLS"

    @classmethod
    def description(cls, key):
        descriptions = {
            cls.FEVER: "A temporary increase in average body temperature of 98.6°F (37°C).",
            cls.COUGH: "A sudden, forceful hacking sound to release air and clear an irritation in the throat or airway.",
            cls.FATIGUE: "A feeling of tiredness or exhaustion or a need to rest because of lack of energy or strength.",
            cls.SHORTNESS_OF_BREATH: "A feeling of not being able to breathe well enough.",
            cls.NAUSEA: "A feeling of sickness with an inclination to vomit.",
            cls.HEADACHE: "Pain in any region of the head, ranging from sharp to dull, that may occur with other symptoms.",
            cls.DIZZINESS: "A sensation of spinning around and losing one's balance.",
            cls.SORE_THROAT: "Pain or irritation in the throat that can occur with or without swallowing.",
            cls.RUNNY_NOSE: "Excess drainage, ranging from a clear fluid to thick mucus, from the nose and nasal passages.",
            cls.MUSCLE_ACHES: "Discomfort or pain in the muscles, often a result of physical activity or illness.",
            cls.LOSS_OF_TASTE: "A reduced or altered ability to taste flavors.",
            cls.LOSS_OF_SMELL: "A reduced or altered ability to detect odors.",
            cls.CHEST_PAIN: "Discomfort or pain that you feel anywhere along the front of your body between your neck and upper abdomen.",
            cls.CHILLS: "Sensation of coldness often accompanied by shivering, typically as a result of fever or illness."
        }
        return descriptions.get(key, "")
class Action(str, Enum):
    HYDRATION = "HYDRATION"
    STRESS_MANAGEMENT = "STRESS_MANAGEMENT"
    SLEEP = "SLEEP"
    EXERCISE = "EXERCISE"
    DIETARY_CHANGES = "DIETARY_CHANGES"
    MEDICATION_ADHERENCE = "MEDICATION_ADHERENCE"
    REGULAR_CHECKUPS = "REGULAR_CHECKUPS"
    AVOID_ALCOHOL = "AVOID_ALCOHOL"
    SMOKING_CESSATION = "SMOKING_CESSATION"
    BALANCED_DIET = "BALANCED_DIET"
    PHYSICAL_THERAPY = "PHYSICAL_THERAPY"
    MENTAL_HEALTH_SUPPORT = "MENTAL_HEALTH_SUPPORT"
    PAIN_MANAGEMENT = "PAIN_MANAGEMENT"
    IMMUNIZATION = "IMMUNIZATION"
    FOLLOW_UP_VISITS = "FOLLOW_UP_VISITS"
    WEIGHT_MANAGEMENT = "WEIGHT_MANAGEMENT"
    BLOOD_PRESSURE_MONITORING = "BLOOD_PRESSURE_MONITORING"
    BLOOD_SUGAR_MONITORING = "BLOOD_SUGAR_MONITORING"
    CHOLESTEROL_CONTROL = "CHOLESTEROL_CONTROL"
    FLUID_INTAKE_MONITORING = "FLUID_INTAKE_MONITORING"

    @classmethod
    def description(cls, key):
        descriptions = {
            cls.HYDRATION: "Drink plenty of water to stay hydrated.",
            cls.STRESS_MANAGEMENT: "Practice relaxation techniques such as deep breathing, meditation, or yoga.",
            cls.SLEEP: "Ensure you get adequate sleep each night to support overall health.",
            cls.EXERCISE: "Engage in regular physical activity to maintain fitness and health.",
            cls.DIETARY_CHANGES: "Adopt a balanced diet to support your health and well-being.",
            cls.MEDICATION_ADHERENCE: "Take your medications as prescribed by your healthcare provider.",
            cls.REGULAR_CHECKUPS: "Schedule regular checkups with your healthcare provider to monitor your health.",
            cls.AVOID_ALCOHOL: "Avoid consuming alcohol to prevent adverse health effects.",
            cls.SMOKING_CESSATION: "Quit smoking to improve your overall health and reduce the risk of disease.",
            cls.BALANCED_DIET: "Maintain a balanced diet to ensure you get all necessary nutrients.",
            cls.PHYSICAL_THERAPY: "Participate in physical therapy sessions to aid in recovery and improve mobility.",
            cls.MENTAL_HEALTH_SUPPORT: "Seek mental health support to manage stress and emotional well-being.",
            cls.PAIN_MANAGEMENT: "Follow pain management strategies to alleviate discomfort.",
            cls.IMMUNIZATION: "Stay up-to-date with immunizations to protect against infectious diseases.",
            cls.FOLLOW_UP_VISITS: "Attend follow-up visits with your healthcare provider to track progress.",
            cls.WEIGHT_MANAGEMENT: "Maintain a healthy weight through diet and exercise.",
            cls.BLOOD_PRESSURE_MONITORING: "Regularly monitor your blood pressure to manage hypertension.",
            cls.BLOOD_SUGAR_MONITORING: "Keep track of your blood sugar levels to manage diabetes.",
            cls.CHOLESTEROL_CONTROL: "Manage your cholesterol levels through diet, exercise, and medication if needed.",
            cls.FLUID_INTAKE_MONITORING: "Monitor your fluid intake to ensure proper hydration.",
        }
        return descriptions.get(key, "")


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Patient(BaseModel):
    name: str = Field(..., description="The first name of the patient")
    surname: str = Field(..., description="The last name of the patient")
    age: int = Field(..., description="The age of the patient, if not specified = 0")
    gender: str = Field(..., description="The gender of the patient. Male of Female")
    address: Address = Field(..., description="The address of the patient")
    phone_number: str = Field(..., description="The phone number of the patient")
    email: str = Field(..., description="The email of the patient")


class TreatmentPlan(BaseModel):
    time_period: int = Field(..., description="The time period for the treatment plan in days, if not specified = 1")
    actions: list[Action]
    medications: list[Medication]


class MedicalReport(BaseModel):
    patient: Patient
    diagnosis: Diagnosis
    treatment_plan: TreatmentPlan
    symptoms: list[Symptom]