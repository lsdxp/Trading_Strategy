import csv


base="/Users/shengdongliu/Trading_Strategy/output_data/"

def get_gold(n=100):
    with open('/Users/shengdongliu/Trading_Strategy/output_data/metal/GOLD.csv') as inf:
          csvr = csv.reader(inf)
          csvr.next()
          prices = [float(row[6]) for row in csvr]
          prices=prices[:n]
          prices.reverse()
    return prices
def get_silver(n=100):
    with open('/Users/shengdongliu/Trading_Strategy/output_data/metal/SLV.csv') as inf:
          csvr = csv.reader(inf)
          csvr.next()
          prices2 = [float(row[6]) for row in csvr]
          prices2=prices2[:n]
          prices2.reverse()
    return prices2
def get_pt(n=100):
    with open('/Users/shengdongliu/Trading_Strategy/output_data/metal/PPLT.csv') as inf:
          csvr = csv.reader(inf)
          csvr.next()
          prices3 = [float(row[6]) for row in csvr]
          prices3=prices3[:n]
          prices3.reverse()
    return prices3

def get_date(n=100):
    with open('/Users/shengdongliu/Trading_Strategy/output_data/metal/GOLD.csv') as inf:
          csvr = csv.reader(inf)
          csvr.next()
          date = [row[0] for row in csvr]
          date=date[:n]
          date.reverse()
    return date