import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def initialize_base_population():
    population_graph = BayesianNetwork(
        [
            ("SES", "X1"), ("G", "X1"),
            ("SES", "X2"), ("G", "X2"),
            ("SES", "X3"), ("G", "X3"),
            ("X1", "Z1"), ("X2", "Z1"), ("X3", "Z1")
        ]
    )

    cpd_G = TabularCPD(variable="G", variable_card=2, values=[[0.4], [0.6]])

    cpd_SES = TabularCPD(variable="SES", variable_card=2, values=[[0.8], [0.2]])

    cpd_X3 = TabularCPD(
        variable="X3",
        variable_card=2,
        values=[[0.8, 0.5, 0.5, 0.2],
                [0.2, 0.5, 0.5, 0.8]
               ],
        evidence=["G", "SES"],
        evidence_card=[2, 2],
    )

    cpd_X2 = TabularCPD(
        variable="X2",
        variable_card=2,
        values=[[0.8, 0.6, 0.2, 0.1],
                [0.2, 0.4, 0.8, 0.9]
               ],
        evidence=["G", "SES"],
        evidence_card=[2, 2],
    )

    cpd_X1 = TabularCPD(
        variable="X1",
        variable_card=2,
        values=[[0.9, 0.2, 0.5, 0.1],
                [0.1, 0.8, 0.5, 0.9]
               ],
        evidence=["G", "SES"],
        evidence_card=[2, 2],
    )

    cpd_Z1 = TabularCPD(
        variable="Z1",
        variable_card=2,
        values=[[1, 0.6, 0.6, 0.2, 0.6, 0.2, 0.2, 0],
                [0, 0.4, 0.4, 0.8, 0.4, 0.8, 0.8, 1]
               ],
        evidence=["X1", "X2", "X3"],
        evidence_card=[2, 2, 2],
    )

    population_graph.add_cpds(cpd_G, cpd_SES, cpd_X1, cpd_X2, cpd_X3, cpd_Z1)
    
    return population_graph

def initialize_BN():

    G_EO = BayesianNetwork(
        [
            ("SEX", "SCHOOL"), ("SES", "SCHOOL"),
            ("SCHOOL", "SAT"), ("SEX", "SAT"), ("SES", "SAT"),
            ("SAT", "COLLEGE"), ("SEX", "COLLEGE"), ("SES", "COLLEGE"),
            ("COLLEGE", "CGPA"), ("SEX", "CGPA"), ("SES", "CGPA"),
            ("COLLEGE", "INTERN"), ("CGPA", "INTERN"), ("SEX", "INTERN"), ("SES", "INTERN"),
            ("INTERN", "JOB"), ("COLLEGE", "JOB"), ("SEX", "JOB"), ("SES", "JOB")
        ]
    )

    cpd_SEX = TabularCPD(variable="SEX", variable_card=2, values=[[0.6], [0.4]])

    cpd_SES = TabularCPD(variable="SES", variable_card=3, values=[[0.2], [0.4], [0.4]])

    cpd_SCHOOL = TabularCPD(
        variable="SCHOOL",
        variable_card=3,
        values=[[0.7, 0.9, 0.4, 0.4, 0.1, 0.2],
                [0.2, 0.07, 0.5, 0.3, 0.3, 0.4],
                [0.1, 0.03, 0.1, 0.3, 0.6, 0.4]
               ],
        evidence=["SEX", "SES"],
        evidence_card=[2, 3],
    )

    cpd_SAT = TabularCPD(
            variable="SAT",
        variable_card=3,
        values=[
            [0.70, 0.4, 0.3, 0.4, 0.3, 0.05, 0.55, 0.3, 0.1, 0.2, 0.4, 0.4, 0.3, 0.2, 0.2, 0.1, 0.05, 0.01],
            [0.25, 0.4, 0.3, 0.4, 0.5, 0.30, 0.30, 0.6, 0.5, 0.7, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.25, 0.19],
            [0.05, 0.2, 0.4, 0.2, 0.2, 0.65, 0.15, 0.1, 0.4, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.70, 0.80]
        ],
        evidence=["SCHOOL", "SEX", "SES"],
        evidence_card=[3, 2, 3],
    )

    cpd_COLLEGE = TabularCPD(
            variable="COLLEGE",
        variable_card=2,
        values=[
            [0.9, 0.7, 0.4, 0.9, 0.7, 0.1, 0.9, 0.6, 0.3, 0.6, 0.2, 0.4, 0.5, 0.2, 0.2, 0.4, 0.15, 0.02],
            [0.1, 0.3, 0.6, 0.1, 0.3, 0.9, 0.1, 0.4, 0.7, 0.4, 0.8, 0.6, 0.5, 0.8, 0.8, 0.6, 0.85, 0.98]
        ],
        evidence=["SAT", "SEX", "SES"],
        evidence_card=[3, 2, 3],
    )

    cpd_CGPA = TabularCPD(
            variable="CGPA",
        variable_card=3,
        values=[
            [0.6, 0.4, 0.2, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05],
            [0.3, 0.5, 0.7, 0.3, 0.2, 0.1, 0.6, 0.7, 0.7, 0.3, 0.4, 0.4],
            [0.1, 0.1, 0.1, 0.4, 0.6, 0.8, 0.1, 0.1, 0.2, 0.6, 0.5, 0.55]
        ],
        evidence=["COLLEGE", "SEX", "SES"],
        evidence_card=[2, 2, 3],
    )

    cpd_INTERN = TabularCPD(
            variable="INTERN",
        variable_card=2,
        values=[
            [0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.5, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.5, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.4, 0.5, 0.8, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        ],
        evidence=["CGPA", "COLLEGE", "SEX", "SES"],
        evidence_card=[3, 2, 2, 3],
    )

    cpd_JOB = TabularCPD(
            variable="JOB",
        variable_card=2,
        values=[
            [0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.9, 0.8, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05],
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.8, 0.8, 0.8, 0.9, 0.95, 0.95]
            ],
        evidence=["INTERN", "COLLEGE", "SEX", "SES"],
        evidence_card=[2, 2, 2, 3],
    )

    G_EO.add_cpds(cpd_SEX, cpd_SES, cpd_SCHOOL, cpd_SAT, cpd_COLLEGE, cpd_CGPA, cpd_INTERN, cpd_JOB)
    
    return G_EO
