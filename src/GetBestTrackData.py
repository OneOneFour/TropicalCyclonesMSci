import csv
from fetch_file import get_data

with open("Data/HURDAT2", mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    hurricane_strength_data = []
    TC_identifiers = []
    num_hurricanes = 0
    for row in csv_reader:
        if row['Date'][0] != '2':
            TC_class = row["Date"]
            TC_name = row['Time'].strip()
            TC_data_points = int(row['Note'].strip())
            TC_identifiers += [[TC_class, TC_name, TC_data_points]]
            num_hurricanes += 1
        else:
            if row['Type'].strip() == 'HU':
                hurricane_strength_data += [[row["Date"], row["Time"], row["Type"], row["Lat"], row["Long"], row["WindMax"], row["MinPres"]]]
                rel_files = get_data(root_dir="Data", year=int(row["Date"][0:4]), month=int(row["Date"][4:6]), day=int(row["Date"][6:8]),
                                    north=int(row["Lat"][1:-3])+1, south=int(row["Lat"][1:-3]) - 1, west=int(row["Long"][1:-3]) - 1, east=int(row["Long"][1:-3]) + 1,
                                    dayOrNight="DNB", mode="best_track_compare")
                file_times = [int(file[23:27]) for file in rel_files if "VNP02" in file]
                for time in file_times:
                    if abs(time - int(row["Time"].strip())) < 300:
                        print("Matching files found at %i and %i" % (time, int(row["Time"].strip())))


#for i in range(len(TC_identifiers)):
#    print(TC_identifiers[i])
#    for j in range(TC_identifiers[i][2]):
#        print(hurricane_strength_data[i+j])
