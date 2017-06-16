# TT-SEMPRE: Semantic Parsing for Compound Virtual Assistant Commands

This repository contains TT-SEMPRE, a fork of [SEMPRE](https://github.com/percyliang/sempre).
TT-SEMPRE is a semantic parser that understands virtual assistant commands
of the form when-get-do

## Installation

Download dependencies:

    ./pull-dependencies core corenlp thingtalk

Build the core:

    JAVAHOME=<path to java> ant thingtalk

Build the HTTP server:

    JAVAHOME=<path to java> ant api

`JAVAHOME` should be set to the path to your Java installation, eg. `/usr/lib/jvm/openjdk-1.8.0`.
Java 1.8 is required. A working C compiler must also be installed.

## Running TT-SEMPRE interactively

    ./run.sh interactive -ThingpediaDatabase.dbPw <DATABASE_PW> -ThingpediaLexicon.subset <SUBSET>

Where `<DATABASE_PW>` is the password to the Thingpedia Database, and `<SUBSET>` is a space separated
list of devices to limit the scope of Thingpedia (e.g. `twitter instagram`).

In interactive mode, it is possible to type sentences and check how they are parsed.

## Training

    ./run-training.sh -ThingpediaDatabase.dbPw <DATABASE_PW> -ThingpediaLexicon.subset <SUBSET>

This command will fetch the data from the Thingpedia dataset and run a full session of training.
The trained model (.params file) will be saved under `./almond`.

You must have the Berkeley Aligner installed. By default the script looks in the parent directory
of the TT-SEMPRE checkout. Set the environment variable `BERKELEYALIGNER` if you installed it
elsewehre.

You can specify the environment variables `TRAINING` and `TESTING` (as a comma separated list of
Thingpedia dataset names, eg `thingpedia,online,turking-prim0`) to control the datasets used
for training and testing. The default covers the normal Thingpedia dataset used to train Almond.

If you specify the environment variables `SEMPRE_USER` and `SEMPRE_SERVER`, the trained model
will be uploaded to the given server (using the given user through ssh) and the server will
be reloaded.

## Running the server

    ./run.sh server -ThingpediaDatabase.dbPw <DATABASE_PW> -ThingpediaLexicon.subset <SUBSET>

The server runs on port 8400 by default. Use `-APIServer.port <X>` and `-APIServer.ssl_port <X>`
to change the ports and to enable TLS.

### Querying the server

    http://127.0.0.1:8400/query?q=<query>&locale=<lang>&limit=<x>

Set `<query>` to the sentence to parse, `<lang>` to the locale code (eg. `en-US`) and `<x>` to the
maximum number of results to report. If limit is unspecified it defaults to 20. If the locale
is unspecified it defaults to `en-US` (American English).

Result:

    {"sessionId":"....",
     "candidates":[
        {"prob":0.5,"score":1,"answer":"..."},
        {"prob":0.5,"score":1,"answer":"..."},
     ]}
