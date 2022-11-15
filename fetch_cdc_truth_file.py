import os

URL = "https://raw.githubusercontent.com/cdcepi/Flusight-forecast-data/master/data-truth/" \
      "truth-Incident%20Hospitalizations.csv"
OUT_FNAME = '"hosp_data/truth-Incident Hospitalizations.csv"'


def main():
    answer = input("You're about to download updated hospitalization data from CDC github.\n"
                   "Did you backup the previous file? (y/n)")

    if answer.lower() != "y":
        print("Then go do it...")
        return

    answer = input("Ok, are you sure you want to download new data and overwrite existing file then? (y/n)")

    if answer.lower() != "y":
        print("Quitting now...")
        return

    os.system(f"curl {URL} -o {OUT_FNAME}")


if __name__ == "__main__":
    main()
