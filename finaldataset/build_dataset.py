import csv


def main():
    # Load In-N-Out locations (label = 1)
    with open("../justin/densitytest.csv", newline="") as f:
        in_n_outs = list(csv.DictReader(f))

    # Load rejected locations (label = 0)
    with open("../justin/rejected_locations.csv", newline="") as f:
        rejected = list(csv.DictReader(f))

    rows = []

    for ino in in_n_outs:
        rows.append({
            "label": 1,
            "name": ino["name"],
            "lat": ino["lat"],
            "lon": ino["lon"],
            "city": ino["city"],
            "state": ino["state"],
            "postcode": ino["postcode"],
            "street": ino["street"],
            "housenumber": ino["housenumber"],
            "start_date": ino.get("start_date", ""),
            "osm_first_seen": ino.get("osm_first_seen", ""),
        })

    for rej in rejected:
        rows.append({
            "label": 0,
            "name": rej["name"],
            "lat": rej["lat"],
            "lon": rej["lon"],
            "city": rej["city"],
            "state": "",
            "postcode": rej["postcode"],
            "street": rej["street"],
            "housenumber": rej["housenumber"],
            "start_date": rej.get("competitor_created", ""),
            "osm_first_seen": rej.get("competitor_created", ""),
        })

    output = "final_dataset.csv"
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    accepted = sum(1 for r in rows if r["label"] == "1" or r["label"] == 1)
    print(f"Wrote {len(rows)} rows ({accepted} accepted, {len(rows) - accepted} rejected) to {output}")


if __name__ == "__main__":
    main()
