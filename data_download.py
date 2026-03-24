from nbiatoolkit import NBIAClient
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Download NBIA series data by collection name")
    parser.add_argument(
        "-c", "--collection",
        required=True,
        help="Name of the NBIA collection to download (e.g., 'NSCLC Radiogenomics')"
    )
    args = parser.parse_args()

    collection_name = args.collection
    with NBIAClient() as nbia:
        print(f"Getting series for collection: {collection_name} ...")
        try:
            data = nbia.getSeries(collection_name)
            if not data:
                print(f"No series found for collection '{collection_name}'")
                sys.exit(0)

            print(f"Found {len(data)} series. Downloading...")
            for file in data:
                seriesInstanceUID = file.get("SeriesInstanceUID")
                print("Downloading {}".format(seriesInstanceUID))
                nbia.downloadSeries(seriesInstanceUID)
            print("Download completed!")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()