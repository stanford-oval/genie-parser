#!/bin/bash

# make sure we run on an UTF-8 locale
export LC_ALL=en_US.utf8

die() {
    echo "$@"
    exit 1
}

set -e

LANGUAGE_TAG="$1"
shift
export LANGUAGE_TAG
test -n "${LANGUAGE_TAG}" || die "language must be specified as an argument to this script"

ADMIN_TOKEN=admin
MODULE=${MODULE:-almond}
SEMPRE_SERVER=${SEMPRE_SERVER:-127.0.0.1}
SEMPRE_PORT=${SEMPRE_PORT:-8400}
SEMPRE_USER=${SEMPRE_USER:-sempre-training}
export MODULE
export ADMIN_TOKEN

set -x

SEMPREDIR=`dirname $0`
SEMPREDIR=`realpath ${SEMPREDIR}`
export SEMPREDIR
WORKDIR=`mktemp -t -d workdir-XXXXXX`
WORKDIR=`realpath "${WORKDIR}"`
export WORKDIR
mkdir ${WORKDIR}/${MODULE}
cd ${SEMPREDIR}

on_error() {
    rm -fr ${WORKDIR}
}
trap on_error ERR INT TERM

# train on everything
TRAINING=${TRAINING:-thingpedia,turking-prim0,turking-prim1,turking-prim2,turking-prim3,turking-compound0,turking-compound1,turking-compound2,turking-compound3,turking-compound4,turking3-compound0,turking3-compound1,turking3-compound2,turking3-compound3,turking3-compound4,turking3-compound5,turking3-compound6,online,online-bookkeeping}
export TRAINING
# test on nothing
TESTING=${TESTING:-}
export TESTING

cp ${SEMPREDIR}/module-classes.txt ${WORKDIR}/

# extract the canonicals from the db
${SEMPREDIR}/scripts/run-download-dataset.sh \
    ${WORKDIR}/${MODULE}/canonicals.${LANGUAGE_TAG}.tsv \
    -ThingpediaDataset.trainTypes $(echo $TRAINING | tr ',' ' ') \
    "$@"

# run the berkeley aligner
${SEMPREDIR}/scripts/run-berkeley-aligner.sh ${WORKDIR}/${MODULE}/canonicals.${LANGUAGE_TAG}.tsv

# actually run sempre
# include everything in training, and skip testing
${SEMPREDIR}/run.sh training \
    -execDir ${WORKDIR}/sempre.tmp \
    -ThingpediaDataset.testTypes $(echo $TESTING | tr ',' ' ') \
	-ThingpediaDataset.trainTypes $(echo $TRAINING | tr ',' ' ' ) \
	"$@"
# copy the final params
cp ${WORKDIR}/sempre.tmp/params.2 ${WORKDIR}/${MODULE}/${MODULE}.${LANGUAGE_TAG}.params

# move the new files to the right place
FILES="${WORKDIR}/${MODULE}/${MODULE}.word_alignments.berkeley.${LANGUAGE_TAG} ${WORKDIR}/${MODULE}/${MODULE}.phrase_alignments.${LANGUAGE_TAG} ${WORKDIR}/${MODULE}/${MODULE}.${LANGUAGE_TAG}.params"
case ${SEMPRE_SERVER} in
	127.0.0.1)
		cp ${FILES} ${SEMPREDIR}/${MODULE}
		;;
	*)
		scp ${FILES} ${SEMPRE_USER}@${SEMPRE_SERVER}:/opt/sempre/${MODULE}
        ;;
esac

# tell the server to reload itself, ignore errors
curl -v -s "https://${SEMPRE_SERVER}:${SEMPRE_PORT}/admin/reload?locale=${LANGUAGE_TAG}&accessToken=${ADMIN_TOKEN}" || true

# clean up
rm -fr ${WORKDIR}
