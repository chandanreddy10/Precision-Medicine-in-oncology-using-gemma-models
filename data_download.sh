#!/bin/bash

DATA_DIR="data"

[ ! -d "$DATA_DIR" ] && mkdir -p "$DATA_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 filename.txt"
    exit 1
fi

file="$1"

[ ! -f "$file" ] && { echo "Error: $file not found"; exit 1; }

while IFS= read -r url || [ -n "$url" ]; do
    url="${url//$'\r'/}"

    [ -z "$url" ] && continue

    echo "Downloading: $url"

    wget -c -P "$DATA_DIR" "$url"
done < "$file"