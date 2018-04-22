#!/usr/bin/env bash

hdfs dfs -rm -r hits_graph&
git fetch
git rebase
gradle jar
hadoop jar build/libs/hits_and_pagerank.jar GraphBuilder /data/infopoisk/hits_pagerank/docs-000.txt hits_graph