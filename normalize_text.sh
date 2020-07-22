#!/usr/bin/env bash

normalize_text() {
  tr '[:upper:]' '[:lower:]' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " "
}
DATADIR=${DATADIR:-data}
set -e

data_result="${DATADIR}/PT_corpus.train"
if [ ! -f "$data_result" ]
then
    cat "${DATADIR}/PT_corpus" | normalize_text > "$data_result" || rm -f "$data_result"
fi
