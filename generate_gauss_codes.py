import requests
import csv
import re
import time
from bs4 import BeautifulSoup

def fetch_gauss_code(knot_name):
    
    url = f"https://katlas.org/wiki/Data:{knot_name}/Gauss_Code"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"No page for {knot_name}")
            return None
    except Exception as e:
        print(f"Error fetching {knot_name}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    
    body_text = soup.get_text(separator="\n")

    
    if match:
        return match.group(0).strip()
    return None


def generate_gauss_csv(output_file="gauss_codes.csv"):
    knots = ["unknot"]
    for n in range(3, 11):   
        for i in range(1, 200):  
            knots.append(f"{n}_{i}")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["knot", "gauss_code"])

        for knot in knots:
            print(f"Fetching {knot}...")
            code = fetch_gauss_code(knot)
            if code:
                writer.writerow([knot, code])
            time.sleep(0.3)  

    print(f"\n Saved Gauss codes to {output_file}")


if __name__ == "__main__":
    generate_gauss_csv()
