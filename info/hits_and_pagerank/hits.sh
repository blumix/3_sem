#!/usr/bin/env bash

hdfs dfs -rm -r hits_out/a_scores_1 &
git fetch
git rebase
gradle jar
hadoop jar build/libs/hits_and_pagerank.jar Hits /user/m.belozerov/hits_graph/part-r-00000
