#!/usr/bin/env bash

hdfs dfs -rm -r hits_graph&
git fetch
git rebase
export LIBJARS=/home/m.belozerov/3_sem/info/hits_and_pagerank/jsoup-1.11.3.jar
export HADOOP_CLASSPATH=/home/m.belozerov/3_sem/info/hits_and_pagerank/jsoup-1.11.3.jar
gradle jar
hadoop jar build/libs/hits_and_pagerank.jar GraphBuilder -libjars ${LIBJARS} /data/infopoisk/hits_pagerank/docs-000.txt hits_graph