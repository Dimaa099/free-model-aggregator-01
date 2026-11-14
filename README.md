# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Dimaa099/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Dimaa099/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

An aggregator server that answers OpenAI- and Anthropic-wire requests with the highest-rated free model across ten providers, translating losslessly between the OpenAI, Anthropic and Gemini wires and failing over whenever a provider refuses. Python 3.13, standard library only; every routing choice derives from the trial brief and the providers' answers, so a replayed trial reproduces the live one byte for byte.

## Run it

Against a practice instance, with the kit's env and scorer (the kit lives beside this repository in the author's workspace):

```sh
KIT=<unpacked free-model-aggregator-01 kit>
INST=$KIT/practice/trial-0
mkdir -p /tmp/agg/state
MAKEFUN_ENV_DATA_DIR=$INST/input/env MAKEFUN_STATE_DIR=/tmp/agg/state MAKEFUN_ENV_PORT=18101 \
  node $KIT/program/env/server.mjs &
MAKEFUN_ENV_URL=http://127.0.0.1:18101 MAKEFUN_PROMPT_FILE=$INST/input/prompt.txt \
  python3 main.py            # registers, serves the 20 requests, exits 0
kill -TERM %1                # the env writes its state export
MAKEFUN_OUTPUT_DIR=/tmp/agg/state/export MAKEFUN_REFERENCE_DIR=$INST/reference \
  MAKEFUN_SCORES_DIR=/tmp/agg/scores node $KIT/program/scorer/score.mjs
```

`python3 main.py --list` prints the catalog document the server registers.

---

<sub>Entered on [make.fun](https://make.fun/?ref=dimarz&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
