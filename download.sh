#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
DATA_DIR="$SCRIPT_DIR/data"

PT_FTP="ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz"

### 2019 MeSH descriptors and supplemental concept records
#echo "- Downloading MeSH descriptors and supplemental concept records"
#DIR_MESH="$DOWNLOAD_PATH/mesh"
#if [ ! -d "$DIR_MESH" ]; then
#    mkdir -p "$DIR_MESH"
#fi
#wget -P "$DIR_MESH" -c "https://mir.jiho.us/bmet/desc2019.gz"
#echo "    + Decompressing gzip files..."
#cd $DIR_MESH
#gzip -d "desc2019.gz"
#cd "$DIR_SCRIPT"

# Download PubTator
echo "Downloading PubTator data file"
if [ ! -d "$DATA_DIR/pubtator" ]; then
    mkdir -p "$DATA_DIR/pubtator"
fi
wget -P "$DATA_DIR/pubtator" -c "$PT_FTP"
