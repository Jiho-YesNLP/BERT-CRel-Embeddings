#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
DATA_DIR="$SCRIPT_DIR/data"

# MeSH descriptor
echo "- Validating MeSH descriptor files"
MESH_FILE="ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/xmlmesh/desc2020.xml"
if [ ! -d "$DATA_DIR/mesh" ]; then
    mkdir -p "$DATA_DIR/mesh"
fi
if [ ! -f "$DATA_DIR/mesh/desc2020.xml" ]; then
    wget -P "$DATA_DIR/mesh" -c "$MESH_FILE"
fi

# Download PubTator
echo "- Validating PubTator data file"
PT_FILE="ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz"
if [ ! -d "$DATA_DIR/pubtator" ]; then
    mkdir -p "$DATA_DIR/pubtator"
fi
if [ ! -f "$DATA_DIR/pubtator/bioconcepts2pubtatorcentral.offset.gz" ]; then
    wget -P "$DATA_DIR/pubtator" -c "$PT_FILE"
fi

# Download PubMed
echo "- Validating PubMed data files"
if [ ! -d "$DATA_DIR/pubmed" ]; then
    mkdir -p "$DATA_DIR/pubmed"
fi
cd "$DATA_DIR/pubmed"
for num in $(seq -f "%04g" 1 1015)
do
    FILENAME1="pubmed20n$num.xml.gz"
    FILENAME2="pubmed20n$num.xml.gz.md5"
    PM_FILE1="ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/$FILENAME1"
    PM_FILE2="ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/$FILENAME2"
    if [ -f "$FILENAME1" ] && md5sum --status -c $FILENAME2; then
        echo "$PM_FILE1 - okay"
    else
        wget -c $PM_FILE1 $PM_FILE2
    fi
done
cd $SCRIPT_DIR
