import csv
class Patient:
    all_patients = []
    def __init__ (self, subject:str, ff_Rbase:float, ff_R1:float, ff_R2:float, ff_Lbase:float, ff_L1:float, ff_L2:float, d4z4:int):
        self.subject = subject
        self.ff_Rbase = ff_Rbase
        self.ff_R1 = ff_R1
        self.ff_R2 = ff_R2 
        self.ff_Lbase = ff_Lbase
        self.ff_L1 = ff_L1
        self.ff_L2 = ff_L2 
        self.d4z4 = d4z4
        Patient.all_patients.append(self)

    def __repr__ (self):
         return f"{self.subject}: ({self.ff_Rbase} | {self.ff_R1} | {self.ff_R2} | {self.ff_Lbase} | {self.ff_L1} | {self.ff_L2}  {self.d4z4})"

    def instantiate_from_csv (cls, filename:str):
         with open(filename, encoding="utf8") as f:
                    reader = csv.DictReader(f)
                    rows_of_patients = list(reader)
                    for row in rows_of_patients:
                        Patient(subject = row['subject_id'], ff_Rbase = float(row['ff_baseline_R']), ff_R1 = float(row['ff_y1_R'].strip()) if row['ff_y1_R'].strip() else None, ff_R2 = float(row['ff_y2_R'].strip()) if row['ff_y2_R'].strip() else None, ff_Lbase = float(row['ff_baseline_L']), ff_L1 = float(row['ff_y1_L'].strip()) if row['ff_y1_L'].strip() else None, ff_L2 = float(row['ff_y2_L'].strip()) if row['ff_y2_L'].strip() else None, d4z4 = int(row['d4z4']))