#!/usr/bin/env bash
hadoop dfs -rm -r /out
gradle jar
hadoop jar build/libs/hw1.jar WordCountJob /11e6c777-38e4-4584-86f5-2e50c4144854.pkz /out