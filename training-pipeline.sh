#!/bin/sh

die() {
    echo "$@"
    exit 1
}

set -e

test -n ${DATABASE_URL} || die "DATABASE_URL must be set";
set -x

LANGUAGE_TAG=${LANGUAGE_TAG:-en}
THINGENGINE=${THINGENGINE:-`pwd`/../thingengine-platform-cloud}

# extract the canonicals from the db
node ${THINGENGINE}/scripts/reconstruct_canonicals.js ./sabrina/sabrina.canonicals.tsv

# run the berkeley aligner with default parameters
./run-berkeley-aligner.sh

# here would optionally clean up the ppdb, but we don't yet

# actually run sempre
export LANGUAGE_TAG
./run-sempre-training.sh
