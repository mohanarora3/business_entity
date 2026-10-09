"""Generate a synthetic dataset in the exact competition layout, for smoke-testing the
pipeline before (or without) the real data. NOT used for the real submission.

    python tools/make_synthetic.py --out /tmp/synth_dataset

Mimics the noise described in the problem statement: abbreviations, legal-suffix changes,
DBA names, typos, word-order swaps, transliteration variants, landmark addresses, missing
PIN/ZIP codes, chains (same name, different branch = NOT a match), singletons, and a
test-only third country (France).
"""

import argparse
import random
from pathlib import Path

R = random.Random(7)

US_WORDS = ["Summit", "Pioneer", "Golden", "Liberty", "Evergreen", "Harbor", "Atlas", "Cedar", "Redwood",
            "Blue Ridge", "Silver Lake", "Prairie", "Keystone", "Granite", "Maple", "Union", "Eagle", "Coastal",
            "Riverside", "Northwind", "Brighton", "Sterling", "Lakeside", "Oakwood", "Frontier", "Heritage"]
US_KIND = ["Logistics", "Dental Care", "Auto Repair", "Bakery", "Consulting", "Hardware", "Pharmacy",
           "Construction", "Insurance Agency", "Grill", "Plumbing", "Realty", "Software", "Fitness"]
US_LEGAL = ["LLC", "Inc", "Corporation", "Co", "Ltd", ""]
US_STREETS = ["Main", "Oak", "Pine", "Maple", "Washington", "Lake", "Hill", "Park", "Elm", "Cedar", "Sunset", "Lincoln"]
US_TYPES = [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr"), ("Lane", "Ln")]
US_CITIES = [("Austin", "TX", "787"), ("Denver", "CO", "802"), ("Columbus", "OH", "432"), ("Seattle", "WA", "981"),
             ("Phoenix", "AZ", "850"), ("Atlanta", "GA", "303"), ("Portland", "OR", "972"), ("Boston", "MA", "021")]

IN_FIRST = ["Shri Ganesh", "Laxmi", "Balaji", "Sai", "Krishna", "Om", "Durga", "Mahalaxmi", "Annapurna", "Jai Hind",
            "New India", "Royal", "Sunrise", "Agarwal", "Gupta", "Sharma", "Patel", "Bharat", "Shiv Shakti", "Vinayak"]
IN_TRANSLIT = {"Shri": "Sri", "Laxmi": "Lakshmi", "Agarwal": "Aggarwal", "Ganesh": "Ganesha", "Krishna": "Krishnaa",
               "Mahalaxmi": "Mahalakshmi", "Vinayak": "Vinayaka", "Shiv": "Siva", "Durga": "Dhurga"}
IN_KIND = ["Traders", "Enterprises", "Textiles", "Steels", "Medicals", "Electronics", "Sweets", "Hardware Store",
           "Motors", "Agencies", "Jewellers", "Constructions", "Exports", "Pharma"]
IN_LEGAL = ["Pvt Ltd", "Private Limited", "LLP", "Ltd", ""]
IN_AREAS = ["MG Road", "Station Road", "Gandhi Nagar", "Sector 15", "Lajpat Nagar", "Civil Lines", "Ashok Vihar",
            "Industrial Area Phase 2", "Shastri Nagar", "Rajendra Nagar", "Model Town", "Sadar Bazar"]
IN_CITIES = [("New Delhi", "Delhi", "110"), ("Mumbai", "Maharashtra", "400"), ("Bengaluru", "Karnataka", "560"),
             ("Jaipur", "Rajasthan", "302"), ("Pune", "Maharashtra", "411"), ("Lucknow", "Uttar Pradesh", "226")]
IN_LANDMARK = ["Near SBI ATM", "Opp. Bus Stand", "Behind Railway Station", "Near Hanuman Mandir", "Opp. City Hospital"]

FR_WORDS = ["Boulangerie", "Atelier", "Garage", "Cabinet", "Pharmacie", "Librairie", "Brasserie", "Fromagerie"]
FR_NAMES = ["du Marché", "Saint-Michel", "des Lilas", "Lefèvre", "Dubois", "de la Gare", "Moreau", "Château"]
FR_LEGAL = ["SARL", "SAS", "SA", "EURL", ""]
FR_STREETS = [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl."), ("Chemin", "Ch.")]
FR_SNAMES = ["Victor Hugo", "de la République", "Jean Jaurès", "Pasteur", "Saint-Honoré", "des Écoles"]
FR_CITIES = [("Paris", "75"), ("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"), ("Nantes", "44")]


def typo(s):
    if len(s) < 5 or R.random() < 0.4:
        return s
    i = R.randrange(1, len(s) - 1)
    op = R.choice(["drop", "swap", "dup"])
    if op == "drop":
        return s[:i] + s[i + 1:]
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]


def make_entity(country, chain_name=None):
    if country == "US":
        base = chain_name or f"{R.choice(US_WORDS)} {R.choice(US_KIND)}"
        legal = R.choice(US_LEGAL)
        st, sab = R.choice(US_TYPES)
        city, state, zp = R.choice(US_CITIES)
        num = R.randint(10, 9999)
        addr = dict(num=str(num), street=f"{R.choice(US_STREETS)} {st}", street_ab=f"{R.choice(US_STREETS)} {sab}",
                    city=city, state=state, postal=f"{zp}{R.randint(10, 99)}", landmark="")
        addr["street_ab"] = addr["street"].replace(st, sab)
    elif country == "India":
        base = chain_name or f"{R.choice(IN_FIRST)} {R.choice(IN_KIND)}"
        legal = R.choice(IN_LEGAL)
        city, state, pin = R.choice(IN_CITIES)
        area = R.choice(IN_AREAS)
        addr = dict(num=f"{R.choice(['', 'Shop No. ', 'Plot ', 'H.No. '])}{R.randint(1, 450)}", street=area,
                    street_ab=area.replace("Road", "Rd").replace("Nagar", "Ngr").replace("Sector", "Sec"),
                    city=city, state=state, postal=f"{pin}{R.randint(1, 99):03d}", landmark=R.choice(IN_LANDMARK))
    else:
        base = chain_name or f"{R.choice(FR_WORDS)} {R.choice(FR_NAMES)}"
        legal = R.choice(FR_LEGAL)
        (st, sab), sname = R.choice(FR_STREETS), R.choice(FR_SNAMES)
        city, dep = R.choice(FR_CITIES)
        addr = dict(num=str(R.randint(1, 180)), street=f"{st} {sname}", street_ab=f"{sab} {sname}",
                    city=city, state="", postal=f"{dep}0{R.randint(10, 99)}", landmark="")
    return dict(country=country, base=base, legal=legal, addr=addr)


def render_name(e, noisy):
    name, legal = e["base"], e["legal"]
    if noisy:
        if e["country"] == "India":
            for k, v in IN_TRANSLIT.items():
                if k in name and R.random() < 0.5:
                    name = name.replace(k, v)
        swaps = {"Corporation": "Corp", "Private Limited": "Pvt. Ltd.", "Pvt Ltd": "Private Limited",
                 "Inc": "Incorporated", "Co": "Company", "LLC": "L.L.C.", "Ltd": "Limited", "SARL": "S.A.R.L."}
        r = R.random()
        if r < 0.25:
            legal = ""
        elif r < 0.5:
            legal = swaps.get(legal, legal)
        name = name.replace(" and ", " & ") if R.random() < 0.5 else name.replace(" & ", " and ")
        if R.random() < 0.3:
            name = typo(name)
        if R.random() < 0.1 and len(name.split()) >= 2:
            w = name.split()
            name = " ".join(w[1:] + w[:1])
        if R.random() < 0.08:
            name = f"{name} DBA {R.choice(US_WORDS)} {R.choice(['Express', 'Outlet', 'Store'])}"
        if R.random() < 0.15:
            name = name.upper()
        if e["country"] == "India" and R.random() < 0.15:
            name = "M/s. " + name
    return f"{name} {legal}".strip()


def render_addr(e, noisy):
    a = e["addr"]
    street, num, postal, landmark, state = a["street"], a["num"], a["postal"], a["landmark"], a["state"]
    if noisy:
        if R.random() < 0.5:
            street = a["street_ab"]
        if R.random() < 0.3:
            postal = ""
        if R.random() < 0.3:
            state = ""
        if R.random() < 0.2:
            street = typo(street)
        if R.random() < 0.15:
            num = ""
    if e["country"] == "US":
        parts = [f"{num} {street}".strip(), a["city"], f"{state} {postal}".strip()]
    elif e["country"] == "India":
        lm = landmark if (not noisy or R.random() < 0.6) else ""
        parts = [num, street, lm, a["city"], state, postal if not (noisy and R.random() < 0.3) else
                 postal[:3] + " " + postal[3:] if postal else ""]
        if noisy and R.random() < 0.2:
            parts = [lm, street, num, a["city"]]
    else:
        parts = [f"{num} {street}".strip(), f"{postal} {a['city']}".strip()]
    return ", ".join(p for p in parts if p)


def build_split(split, countries, n_entities, start_ids):
    ents = []
    for _ in range(n_entities):
        c = R.choice(countries)
        if R.random() < 0.12 and ents:     # chain: reuse a brand name, different branch
            same = [e for e in ents if e["country"] == c]
            ents.append(make_entity(c, R.choice(same)["base"] if same else None))
        else:
            ents.append(make_entity(c))
    s1, s2, s3, gt = [], [], [], []
    c1, c2, c3 = start_ids
    for e in ents:
        in_s1 = R.random() < 0.75
        n2 = R.choices([0, 1, 2], [0.45, 0.45, 0.10])[0]
        n3 = R.choices([0, 1, 2], [0.5, 0.42, 0.08])[0]
        rec2 = [(render_name(e, True), render_addr(e, True), e["country"]) for _ in range(n2)]
        rec3 = [(render_name(e, True), render_addr(e, True), e["country"]) for _ in range(n3)]
        ids2, ids3 = [], []
        for r in rec2:
            c2 += 1
            ids2.append(f"S2-{c2:05d}")
            s2.append((ids2[-1], *r))
        for r in rec3:
            c3 += 1
            ids3.append(f"S3-{c3:05d}")
            s3.append((ids3[-1], *r))
        if in_s1:
            c1 += 1
            sid = f"S1-{c1:05d}"
            s1.append((sid, render_name(e, R.random() < 0.3), render_addr(e, R.random() < 0.3), e["country"]))
            gt.append((sid, ",".join(ids2 + ids3)))
    for lst in (s1, s2, s3):
        R.shuffle(lst)
    return s1, s2, s3, gt


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def main(out, n_train, n_test):
    out = Path(out)
    H = ["entity_id", "business_name", "business_address", "country"]
    s1, s2, s3, gt = build_split("train", ["US", "India"], n_train, (0, 0, 0))
    write(out / "train/train_source1.tsv", H, s1)
    write(out / "train/train_source2.tsv", H, s2)
    write(out / "train/train_source3.tsv", H, s3)
    write(out / "train/train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    s1, s2, s3, gt = build_split("test", ["US", "India", "France"], n_test, (50000, 50000, 50000))
    write(out / "test/test_source1.tsv", H, s1)
    write(out / "test/test_source2.tsv", H, s2)
    write(out / "test/test_source3.tsv", H, s3)
    write(out / "test_hidden_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    print(f"synthetic dataset written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/synth_dataset")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    a = ap.parse_args()
    main(a.out, a.n_train, a.n_test)
