import pandas as pd
import numpy as np


def counting_total_LLP(df):
    """
    Count the total number of LLPs in the df.
    """
    i = df['LLP_number_per_ev'].iloc[-1]
    return i


def counting_total_events_per_file(csv_file):
    """
    Count the total number of events in the CSV file.
    """
    df = pd.read_csv(csv_file)
    i = df['total_events'].iloc[-1]
    return i


def counting_total_events_produced_LLP(csv_file):
    """
    Count the total number of events which have produced LLP.
    """
    df = pd.read_csv(csv_file)
    i = df['number_of_production'].iloc[-1]
    return i


def calculate_cross_section_file(csv_file):
    """
    Calculate the cross section based on the number of events and the number of LLPs.
    """
    df = pd.read_csv(csv_file)
    total_events = df['total_events'].iloc[-1]
    total_LLPs = df['LLP_number_per_ev'].iloc[-1]
    cross_section = total_LLPs / total_events
    return cross_section



def calculate_cross_section(df):
    total_events = df['total_events'].iloc[-1]
    total_LLPs = df['LLP_number_per_ev'].iloc[-1]
    cross_section = total_LLPs / total_events
    return cross_section


