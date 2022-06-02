from datetime import datetime

def get_date_time(date_string:str):
    d = [int(x) for x in date_string.split(',')] 
    return datetime(d[0],d[1],d[2],d[3],d[4],d[5]) 
