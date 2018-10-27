#!/bin/bash

set -e
set -o pipefail
set -x

srcdir=`dirname $0`
ssh -o IdentityFile=$srcdir/id_rsa.autodeploy autodeploy@almond-training.stanford.edu
ssh -o IdentityFile=$srcdir/id_rsa.autodeploy autodeploy@almond-nl.stanford.edu
