#imported import libraries
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#created Patient class
class Patient:
    #list that holds all patient
    all_patients = []
    #list of attributes each patient object has
    def __init__ (self, subject:str, ff_Rbase:float, ff_R1:float, ff_R2:float, ff_Lbase:float, ff_L1:float, ff_L2:float, y0:float, y1:float, y2:float, d4z4:int):
        self.subject = subject
        self.ff_Rbase = ff_Rbase
        self.ff_R1 = ff_R1
        self.ff_R2 = ff_R2 
        self.ff_Lbase = ff_Lbase
        self.ff_L1 = ff_L1
        self.ff_L2 = ff_L2 
        self.y0 = y0
        self.y1 = y1
        self.y2 = y2
        self.d4z4 = d4z4
        #adds newly created patients to the all_patients list
        Patient.all_patients.append(self)

    #representer method
    def __repr__ (self):
         return f"{self.subject}: ({self.ff_Rbase} | {self.ff_R1} | {self.ff_R2} | {self.ff_Lbase} | {self.ff_L1} | {self.ff_L2} | {self.d4z4})"
    #instatiating method to create patients by reading the csv file/dataset
    def instantiate_from_csv (cls, filename:str):
        with open(filename, encoding="utf8") as f:
            reader = csv.DictReader(f)
            rows_of_patients = list(reader)
            for row in rows_of_patients:
                Patient(subject = row['subject_id'], 
                    ff_Rbase = float(row['ff_baseline_R']),
                    ff_R1 = float(row['ff_y1_R'].strip()) if row['ff_y1_R'].strip() else None,
                    ff_R2 = float(row['ff_y2_R'].strip()) if row['ff_y2_R'].strip() else None, 
                    ff_Lbase = float(row['ff_baseline_L']), 
                    ff_L1 = float(row['ff_y1_L'].strip()) if row['ff_y1_L'].strip() else None, 
                    ff_L2 = float(row['ff_y2_L'].strip()) if row['ff_y2_L'].strip() else None, 
                    y0 = float(row['lag_months_baseline']), 
                    y1 = float(row['lag_months_y1'].strip()) if row['lag_months_y1'].strip() else None, 
                    y2 = float(row['lag_months_y2'].strip()) if row['lag_months_y2'].strip() else None,
                    d4z4 = int(row['d4z4'])
                    )
    
    #baseline year getter
    def get_y0(self):
        return self.y0

    #year 1 getter
    def get_y1(self):
        if self.y1 == None:
            return 0.0
        return self.y1

    #year 2 getter
    def get_y2(self):
        if self.y2 == None:
            return 0.0
        return self.y2

    #right leg baseline fat fraction getter
    def get_ff_Rbase(self):
        return self.ff_Rbase

    #right leg year 1 fat fraction getter
    def get_ff_R1(self):
        if self.ff_R1 == None:
            return 0.0
        return self.ff_R1

    #right leg year 2 fat fraction getter
    def get_ff_R2(self):
        if self.ff_R2 == None:
            return 0.0
        return self.ff_R2

    #d4z4 gene copies getter
    def get_d4z4(self):
        return self.d4z4

    #linear interpolation method
    @classmethod
    def interpolate(cls, patient: Patient, right_leg: bool, T1: float, T2: float):

        #determines if the fat fraction data will pull from the right or left leg columns
        if right_leg:
            ff_values = [patient.ff_Rbase, patient.ff_R1, patient.ff_R2]

        else:
            ff_values = [patient.ff_Lbase, patient.ff_L1, patient.ff_L2]

        #the timepoints that each patient had been tested
        lag_months = [patient.y0, patient.y1, patient.y2]
        #pairs time points and fat fraction for interpolation
        measured_points = [
            (time, value)
            for time, value in zip(lag_months, ff_values)
            if time is not None and value is not None
        ]

        #calculates the linear interpolation values
        times, values = zip(*measured_points)
        f_linear = interp1d(times, values, kind='linear')
        
        return [float(f_linear(T1)), float(f_linear(T2))]

    #finds the largest shared year 1 data point
    @classmethod
    def find_min_y1(cls, list:list):
        min = list[0].get_y1()
        compare = 0.0
        #compares all values in the list provided
        for patient in list:
            compare = patient.get_y1()
            if compare < min:
                min = compare
        return min

    #finds the largest shared year 2 data point
    @classmethod
    def find_min_y2(cls, list:list):
        min = list[0].get_y2()
        compare = 0.0
        #compares all values in the list provided
        for patient in list:
            compare = patient.get_y2()
            if compare < min:
                min = compare
        return min